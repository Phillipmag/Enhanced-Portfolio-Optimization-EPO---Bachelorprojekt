import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

# 1. Indlæsning og forberedelse af data


# Definer C25-tickers (uden Sydbank)
c25_tickers = [
    "NOVO-B.CO", "DSV.CO", "MAERSK-B.CO", "MAERSK-A.CO", "VWS.CO", 
    "DANSKE.CO", "ORSTED.CO", "CARL-B.CO", "COLO-B.CO", "GMAB.CO",
    "PNDORA.CO", "TRYG.CO", "NSIS-B.CO", "DEMANT.CO", "JYSK.CO", 
    "RBREW.CO", "ISS.CO", "AMBU-B.CO", "GN.CO", "ROCK-B.CO", 
    "FLS.CO", "NKT.CO", "BAVA.CO", "ZEAL.CO"
]

print("Henter aktiedata fra Yahoo Finance...")
raw_data = yf.download(c25_tickers, start="2016-01-01", end="2026-04-14", auto_adjust=True)

# Vælger lukkekurserne fra YF
prices = raw_data['Close'].copy()

print("Indlæser lokal data for Sydbank fra Bloomberg (Excel)...")
# Læs Bloomberg Excel-filen (fanen med udbytte-afkast)
syd_df = pd.read_excel("ALSYDB_DC_EQUITY.xlsx", sheet_name="DAY_TO_DAY_TOT_RETURN_GROSS_DVD")

# Sørg for at 'Date' er i datetime-format og sæt som index
syd_df['Date'] = pd.to_datetime(syd_df['Date'])
syd_df.set_index('Date', inplace=True)

# Bloomberg angiver afkast i procent
# Divider med 100 for at få decimalform:
syd_returns_daily = syd_df['Return_%'] / 100

# Skab en "syntetisk" aktiekurs der starter i indeks 100, så det matcher formatet fra YF
syd_prices = (1 + syd_returns_daily).cumprod() * 100
syd_prices.name = "SYDBANK.CO"

# Flet Sydbank-kurserne sammen med C25-kurserne
prices = prices.join(syd_prices, how='outer')

# Der kan være små forskelle i handelsdage mellem Bloomberg og YF.'forward-filler' for at undgå kunstige NaN-huller
prices = prices.ffill()

# Vælger lukkekurser i slutningen af måneden og beregner procentvist afkast 
returns_monthly = prices.resample('ME').last().pct_change().dropna()

print("Henter data for risikofri rente...")
rf_data = yf.download("^IRX", start="2016-01-01", end="2026-04-14")['Close']

# Kontrol: Sørg for at rf_data er en 1D-liste (Series) og ikke en 2D-tabel (DataFrame)
if isinstance(rf_data, pd.DataFrame):
    rf_data = rf_data.iloc[:, 0]

rf_monthly_annualized = rf_data.resample('ME').last()
rf_monthly = (rf_monthly_annualized / 100) / 12  # Månedlig rente i decimaler

# Sørger for at indeks matcher præcist
returns_monthly.index = returns_monthly.index.to_period('M')
rf_monthly.index = rf_monthly.index.to_period('M')

# Fletter aktier og risikofri rente sammen
data = pd.concat([returns_monthly, rf_monthly.rename("RF")], axis=1)

# Kontrol: Tving alle kolonner til at være tal
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Omstrukturer tilbage til DatetimeIndex for plottets skyld
data.index = data.index.to_timestamp(how='end')

# Opsplit data
rf_monthly_aligned = data["RF"].copy()
excess_returns_monthly = data.drop(columns=["RF"]).copy()

# Beregn merafkast (Afkast - RF)
for col in excess_returns_monthly.columns:
    excess_returns_monthly[col] = excess_returns_monthly[col] - rf_monthly_aligned

print("Databehandling afsluttet. Starter backtesting...\n")

# 2. Definition af parametre

lookback_signal = 12 
lookback_cov = 60 # Kræver 60 måneders (5 års) data før første handel
base_corr_shrink = 0.05 
gamma = 250 

w_grid = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# 3. Hjælpefunktioner

#Defination af XSMOM signalet 
def xsmom_signal(window_returns: pd.DataFrame) -> pd.Series:
    cumret = (1 + window_returns).prod() - 1
    s = cumret - cumret.mean()
    pos = s > 0
    neg = s < 0
    if pos.any():
        pos_sum = s[pos].sum()
        if pos_sum != 0: s.loc[pos] = s.loc[pos] / pos_sum
    if neg.any():
        neg_sum = np.abs(s[neg]).sum()
        if neg_sum != 0: s.loc[neg] = -np.abs(s.loc[neg]) / neg_sum
    return s.fillna(0.0)

def shrink_correlation(corr: np.ndarray, shrink: float) -> np.ndarray:
    n = corr.shape[0]
    return (1 - shrink) * corr + shrink * np.eye(n)

def covariance_from_corr_and_vol(corr: np.ndarray, vol: np.ndarray) -> np.ndarray:
    D = np.diag(vol)
    return D @ corr @ D

def raw_optimizer_weights(cov_matrix: np.ndarray, signal: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_cov = np.linalg.pinv(cov_matrix)
    w = (1 / gamma) * (inv_cov @ signal)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w

def summarize_performance(portfolio_returns: pd.Series, name: str):
    portfolio_returns = pd.Series(portfolio_returns).dropna()
    
    # Beregning af årligt merafkast, årlig volatilitet og Sharpe Ratio
    ann_excess_return = portfolio_returns.mean() * 12
    ann_vol = portfolio_returns.std(ddof=1) * np.sqrt(12)
    sharpe = (ann_excess_return / ann_vol) if ann_vol != 0 else np.nan
    
    # Beregning af Maximum Drawdown (MDD) 
    #Skaber et formueindeks (Wealth Index) ved at akkumulere afkastene
    wealth_index = (1 + portfolio_returns).cumprod()
    
    #Finder det højeste historiske punkt frem til enhver given dato (Running Max)
    previous_peaks = wealth_index.cummax()
    
    #Beregner faldet i procent fra toppen for hver måned
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # 4. MDD er det absolut største fald i hele perioden
    mdd = drawdowns.min()

    
    print(f"\n--- {name} ---")
    print(f"Årligt merafkast:      {ann_excess_return*100:.2f}%")
    print(f"Årlig volatilitet:     {ann_vol*100:.2f}%")
    print(f"Sharpe ratio:          {sharpe:.4f}")
    print(f"Max Drawdown:          {mdd*100:.2f}%")


# 4. Dynamisk backtest-funktion for en given w-værdi


def run_backtest_for_w(w_fixed: float, excess_returns_monthly: pd.DataFrame):
    dates, portfolio_returns, gross_exposures = [], [], []
    weight_history = []
    start_idx = max(lookback_signal, lookback_cov)

    for t in range(start_idx, len(excess_returns_monthly)):
        window = excess_returns_monthly.iloc[t - lookback_cov:t]
        available_assets = window.columns[window.notnull().all()]
        
        if len(available_assets) < 2: continue

        signal_window = excess_returns_monthly[available_assets].iloc[t - lookback_signal:t]
        cov_window = excess_returns_monthly[available_assets].iloc[t - lookback_cov:t]
        next_ret = excess_returns_monthly[available_assets].iloc[t]
        next_date = excess_returns_monthly.index[t]
        
        s_t = xsmom_signal(signal_window)
        vol = cov_window.std(ddof=1).values
        corr = cov_window.corr().values

        corr_base = shrink_correlation(corr, base_corr_shrink)
        corr_epo = shrink_correlation(corr_base, w_fixed)
        cov_epo = covariance_from_corr_and_vol(corr_epo, vol)

        weights = raw_optimizer_weights(cov_epo, s_t.values, gamma=gamma)

        dates.append(next_date)
        portfolio_returns.append(float(next_ret.values @ weights))
        gross_exposures.append(np.sum(np.abs(weights)))
        # Gemmer vægtene for den givne måned
        weight_history.append(pd.Series(weights, index=available_assets))

    # Konverterer listen af vægte til en samlet DataFrame
    weights_df = pd.DataFrame(weight_history, index=dates)

    return pd.Series(portfolio_returns, index=dates), pd.Series(gross_exposures, index=dates), weights_df

def run_benchmarks(excess_returns_monthly: pd.DataFrame):
    dates, ret_1n, ret_1sigma, ret_mom = [], [], [], []
    start_idx = max(lookback_signal, lookback_cov)
    
    for t in range(start_idx, len(excess_returns_monthly)):
        window = excess_returns_monthly.iloc[t - lookback_cov:t]
        available_assets = window.columns[window.notnull().all()]
        
        if len(available_assets) < 2: continue

 # Investerer i måned t 
        next_ret = excess_returns_monthly[available_assets].iloc[t]
        next_date = excess_returns_monthly.index[t]

        #Benchmarks:
        
        # 1/n
        n = len(available_assets)
        w_1n = np.repeat(1/n, n)
        
        # 1/sigma
        vols = window[available_assets].std(ddof=1).values
        inv_vols = np.divide(1.0, vols, out=np.zeros_like(vols), where=vols!=0)
        w_1sigma = inv_vols / np.sum(inv_vols)

        # Simple Momentum (Top 5 da vi kun har 25 aktier)
        mom_window = excess_returns_monthly[available_assets].iloc[t - 12:t]
        cum_ret = (1 + mom_window).prod() - 1
        top_5 = cum_ret.nlargest(5).index
        w_mom = pd.Series(0.0, index=available_assets)
        w_mom[top_5] = 1/5 

        ret_1n.append(float(next_ret.values @ w_1n))
        ret_1sigma.append(float(next_ret.values @ w_1sigma))
        ret_mom.append(float(next_ret.values @ w_mom.values))
        dates.append(next_date)
        
    return pd.Series(ret_1n, index=dates), pd.Series(ret_1sigma, index=dates), pd.Series(ret_mom, index=dates)


# 5. Kørsel af backtest for alle w-værdier og benchmarks


all_results, all_gross_exposures, all_weights = {}, {}, {}

for w in w_grid:
    rets, gross, weights_df = run_backtest_for_w(w, excess_returns_monthly)
    all_results[w] = rets
    all_gross_exposures[w] = gross
    all_weights[w] = weights_df  

bench_1n, bench_1sigma, bench_mom = run_benchmarks(excess_returns_monthly)

# 6. Udskriver resultater


for w in w_grid:
    name = "Klassisk MVO (w = 0.00)" if w == 0 else f"EPO (w = {w:.2f})"
    summarize_performance(all_results[w], name)

summarize_performance(bench_1n, "Benchmark: 1/n (Equally Weighted)")
summarize_performance(bench_1sigma, "Benchmark: 1/sigma (Risk Parity)")
summarize_performance(bench_mom, "Benchmark: Simple Momentum (Top 5) - Long only")


# 6.1 Statistik for out-of-sample perioden (top 3 og bund 3 aktier baseret på årligt merafkast i procent)

# Trækker data for out-of-sample perioden automatisk
start_dato_oos = all_weights[0.00].index[0]
slut_dato_oos = all_weights[0.00].index[-1]

oos_returns = excess_returns_monthly.loc[start_dato_oos:slut_dato_oos]

# Beregner det årlige gennemsnitlige afkast i procent for hvert enkelt aktiv
annualized_returns = oos_returns.mean() * 12 * 100

# Beregner det samlede gennemsnit for hele markedet ---
market_average = annualized_returns.mean()

# Finder Top 3 og Bund 3
top_3 = annualized_returns.nlargest(3)
bottom_3 = annualized_returns.nsmallest(3)

print(f"\n{'='*50}")
print(f" Deskriptiv Statistik ({start_dato_oos.date()} til {slut_dato_oos.date()})")
print(f"{'='*50}")

# Printer markedsgennemsnittet her:
print(f" Gennemsnitligt merafkast (Alle aktiver): {market_average:.2f} %\n")

print(" TOP 3 (Årligt merafkast):")
for ind, val in top_3.items():
    print(f"    {ind:<10}: {val:>6.2f} %")

print("\n BUND 3 (Årligt merafkast):")
for ind, val in bottom_3.items():
    print(f"    {ind:<10}: {val:>6.2f} %")
print("="*50 + "\n")

# 6. Plot graf over bruttoeksponering 

fig, ax = plt.subplots(figsize=(12, 6))

plot_grid = [0.00, 0.20, 0.40, 0.70, 1.00] 

# Plotter MVO og EPO modeller
for w in plot_grid:
    if w in all_gross_exposures:
        # Giver pæne navne
        label_name = "Klassisk MVO (w = 0.00)" if w == 0 else f"EPO (w = {w:.2f})"
        
        # Plotter med en lille smule gennemsigtighed for at se linjerne krydse
        ax.plot(all_gross_exposures[w].index, all_gross_exposures[w].values, 
                label=label_name, linewidth=1.5, alpha=0.9)

# Plotter alle Benchmarks (Da de alle altid er ugearede = 1.0)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
           label="Alle Benchmarks (Ugearet = 1.0)")

# Formatering og tekst
ax.set_title(r"Gross Exposure over tid ($\sum |x_i|$) - MVO & EPO vs. Benchmarks", fontsize=14, fontweight='bold')
ax.set_ylabel("Gross Exposure (Gearing)")
ax.set_xlabel("År")

# Kontrol over tidsinterval og format på x-aksen
ax.xaxis.set_major_locator(mdates.YearLocator(1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
plt.xticks(rotation=0) 


# Placerer legend øverst til venstre
ax.legend(title="Strategier & Benchmarks", loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()



# 8. Plot af Maximum Drawdown over tid for MVO, EPO og Benchmarks


fig, ax = plt.subplots(figsize=(12, 6))

#  MVO & EPO
plot_grid = [0.00, 0.20, 0.70, 1.00] 

for w in plot_grid:
    if w in all_results:
        portfolio_returns = all_results[w].dropna()
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        label_name = "Klassisk MVO (w = 0.00)" if w == 0 else f"EPO (w = {w:.2f})"
        
        # Fuldt optrukne linjer til egne modeller
        ax.plot(drawdowns.index, drawdowns.values * 100, 
                label=label_name, linewidth=1.5, alpha=0.9)


# Plot af drawdown for benchmarks
benchmarks = {
    "Bench: 1/n (Equally Weighted)": bench_1n,
    "Bench: 1/sigma (Risk Parity)": bench_1sigma,
    "Bench: Momentum (Top 5)": bench_mom
}

for bench_name, bench_returns in benchmarks.items():
    portfolio_returns = bench_returns.dropna()
    wealth_index = (1 + portfolio_returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Stiplede linjer til benchmarks for at adskille dem visuelt fra EPO-modellerne
    ax.plot(drawdowns.index, drawdowns.values * 100, 
            label=bench_name, linewidth=1.5, linestyle='--', alpha=0.8)


# Tekst og formatering
ax.set_title("Maximum Drawdown over tid: MVO & EPO vs. Benchmarks", fontsize=14, fontweight='bold')
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("År")

# Tvinger x-aksen til 1-års intervaller og vandret tekst (som tidligere aftalt)
ax.xaxis.set_major_locator(mdates.YearLocator(1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
plt.xticks(rotation=0) 

# Tekstboks (legend) placeret i venstre side med en samlet titel
ax.legend(title="Strategier & Benchmarks", loc='lower left', fontsize=10) 
ax.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.show()

