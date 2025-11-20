"""
Trading Signals Web Application
A web-based interface for the multi-factor trading system

Install: pip install streamlit yfinance pandas fredapi
Run: streamlit run trading_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from fredapi import Fred
import json

# Page config
st.set_page_config(
    page_title="Trading Signal System",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

DATABASE = 'trading_signals.db'

class StreamlitTradingSystem:
    def __init__(self, fred_key):
        self.fred_key = fred_key
        self.db = sqlite3.connect(DATABASE, check_same_thread=False)
        self.setup_database()
    
    def setup_database(self):
        cursor = self.db.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date TEXT,
                close REAL,
                volume INTEGER,
                rsi REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS congressional_trades (
                ticker TEXT,
                date TEXT,
                politician TEXT,
                transaction_type TEXT,
                amount TEXT,
                PRIMARY KEY (ticker, date, politician)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment (
                ticker TEXT,
                date TEXT,
                mentions INTEGER,
                sentiment_score REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_indicators (
                date TEXT PRIMARY KEY,
                gdp_growth REAL,
                unemployment REAL,
                vix REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                ticker TEXT,
                date TEXT,
                score REAL,
                factors TEXT,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        self.db.commit()
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def collect_stock_data(self, ticker, period='3mo'):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            hist['RSI'] = self.calculate_rsi(hist['Close'])
            
            cursor = self.db.cursor()
            for date, row in hist.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_prices 
                    (ticker, date, close, volume, rsi)
                    VALUES (?, ?, ?, ?, ?)
                """, (ticker, date.strftime('%Y-%m-%d'), row['Close'], 
                      row['Volume'], row['RSI']))
            
            self.db.commit()
            return hist
            
        except Exception as e:
            st.error(f"Error collecting data for {ticker}: {e}")
            return None
    
    def collect_macro_data(self):
        if not self.fred_key or self.fred_key == 'YOUR_FRED_API_KEY':
            cursor = self.db.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                INSERT OR REPLACE INTO macro_indicators 
                (date, gdp_growth, unemployment, vix)
                VALUES (?, ?, ?, ?)
            """, (today, 2.5, 4.2, 15.3))
            self.db.commit()
            return
        
        try:
            fred = Fred(api_key=self.fred_key)
            vix = fred.get_series('VIXCLS', observation_start=datetime.now() - timedelta(days=30))
            unemployment = fred.get_series('UNRATE', observation_start=datetime.now() - timedelta(days=90))
            gdp = fred.get_series('A191RL1Q225SBEA', observation_start=datetime.now() - timedelta(days=365))
            
            cursor = self.db.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                INSERT OR REPLACE INTO macro_indicators 
                (date, gdp_growth, unemployment, vix)
                VALUES (?, ?, ?, ?)
            """, (today, float(gdp.iloc[-1]), float(unemployment.iloc[-1]), 
                  float(vix.iloc[-1])))
            
            self.db.commit()
            
        except Exception as e:
            st.warning(f"Could not fetch macro data: {e}")
    
    def simulate_congressional_trades(self):
        simulated_trades = [
            ('NVDA', '2024-11-10', 'Sen. Example', 'Purchase', '$50k-100k'),
            ('MSFT', '2024-11-12', 'Rep. Demo', 'Purchase', '$15k-50k'),
            ('AAPL', '2024-11-14', 'Sen. Sample', 'Sale', '$100k-250k'),
        ]
        
        cursor = self.db.cursor()
        for trade in simulated_trades:
            cursor.execute("""
                INSERT OR REPLACE INTO congressional_trades 
                (ticker, date, politician, transaction_type, amount)
                VALUES (?, ?, ?, ?, ?)
            """, trade)
        
        self.db.commit()
    
    def simulate_sentiment_data(self, tickers):
        cursor = self.db.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        
        for ticker in tickers:
            import random
            mentions = random.randint(50, 500)
            sentiment = random.uniform(-1, 1)
            
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment 
                (ticker, date, mentions, sentiment_score)
                VALUES (?, ?, ?, ?)
            """, (ticker, today, mentions, sentiment))
        
        self.db.commit()
    
    def calculate_score(self, ticker):
        score = 0
        factors = []
        
        cursor = self.db.cursor()
        
        # RSI
        cursor.execute("""
            SELECT rsi FROM stock_prices 
            WHERE ticker = ? 
            ORDER BY date DESC LIMIT 1
        """, (ticker,))
        
        result = cursor.fetchone()
        if result and result[0]:
            rsi = result[0]
            if rsi < 30:
                score += 3
                factors.append(f"🟢 RSI Oversold ({rsi:.1f})")
            elif rsi > 70:
                score -= 2
                factors.append(f"🔴 RSI Overbought ({rsi:.1f})")
        
        # Congressional trades
        cursor.execute("""
            SELECT COUNT(*) FROM congressional_trades 
            WHERE ticker = ? 
            AND transaction_type = 'Purchase'
            AND date > date('now', '-7 days')
        """, (ticker,))
        
        trades = cursor.fetchone()[0]
        if trades > 0:
            score += 3
            factors.append(f"🏛️ Congressional Buys ({trades})")
        
        # Sentiment
        cursor.execute("""
            SELECT sentiment_score, mentions FROM sentiment 
            WHERE ticker = ? 
            ORDER BY date DESC LIMIT 1
        """, (ticker,))
        
        result = cursor.fetchone()
        if result:
            sentiment, mentions = result
            if sentiment > 0.5 and mentions > 200:
                score += 2
                factors.append(f"📈 Bullish Sentiment ({sentiment:.2f})")
            elif sentiment < -0.5 and mentions > 200:
                score -= 2
                factors.append(f"📉 Bearish Sentiment ({sentiment:.2f})")
        
        # Volume
        cursor.execute("""
            SELECT volume FROM stock_prices 
            WHERE ticker = ? 
            ORDER BY date DESC LIMIT 5
        """, (ticker,))
        
        volumes = [row[0] for row in cursor.fetchall()]
        if len(volumes) >= 5:
            avg_volume = sum(volumes[1:]) / 4
            if volumes[0] > avg_volume * 1.5:
                score += 2
                factors.append("📊 Volume Spike")
        
        # Macro
        cursor.execute("""
            SELECT vix FROM macro_indicators 
            ORDER BY date DESC LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result:
            vix = result[0]
            if vix < 20:
                score += 1
                factors.append(f"🌤️ Low Volatility (VIX: {vix:.1f})")
            elif vix > 30:
                score -= 1
                factors.append(f"⚠️ High Volatility (VIX: {vix:.1f})")
        
        return score, factors
    
    def generate_signals(self, tickers):
        signals = []
        
        for ticker in tickers:
            score, factors = self.calculate_score(ticker)
            
            if score > 0:
                signals.append({
                    'ticker': ticker,
                    'score': score,
                    'factors': factors
                })
                
                cursor = self.db.cursor()
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    INSERT OR REPLACE INTO signals 
                    (ticker, date, score, factors)
                    VALUES (?, ?, ?, ?)
                """, (ticker, today, score, json.dumps(factors)))
        
        self.db.commit()
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        return signals


def main():
    st.title("📈 Multi-Factor Trading Signal System")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        fred_key = st.text_input(
            "FRED API Key", 
            type="password",
            help="Get your free key from https://fred.stlouisfed.org"
        )
        
        st.markdown("---")
        
        default_tickers = "AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META,AMZN"
        watchlist_input = st.text_area(
            "Watchlist (comma-separated)",
            value=default_tickers,
            height=100
        )
        
        watchlist = [t.strip().upper() for t in watchlist_input.split(",")]
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_analysis:
        if not fred_key:
            st.warning("⚠️ No FRED API key provided. Using dummy macro data.")
        
        system = StreamlitTradingSystem(fred_key)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Collect data
        status_text.text("📊 Collecting market data...")
        total_steps = len(watchlist) + 3
        
        for i, ticker in enumerate(watchlist):
            system.collect_stock_data(ticker)
            progress_bar.progress((i + 1) / total_steps)
        
        status_text.text("🌍 Collecting macro data...")
        system.collect_macro_data()
        progress_bar.progress((len(watchlist) + 1) / total_steps)
        
        status_text.text("🏛️ Collecting alternative data...")
        system.simulate_congressional_trades()
        progress_bar.progress((len(watchlist) + 2) / total_steps)
        
        system.simulate_sentiment_data(watchlist)
        progress_bar.progress(1.0)
        
        status_text.text("✅ Generating signals...")
        
        # Generate signals
        signals = system.generate_signals(watchlist)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("## 🎯 Trading Signals")
        
        if not signals:
            st.info("No signals generated. Try adjusting your watchlist.")
        else:
            # Top signal highlight
            top_signal = signals[0]
            st.success(f"**Top Signal: {top_signal['ticker']}** (Score: {top_signal['score']:.1f}/10)")
            
            # Create columns for top 3
            if len(signals) >= 3:
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    if i < len(signals):
                        sig = signals[i]
                        with col:
                            st.markdown(f"### {sig['ticker']}")
                            st.metric("Score", f"{sig['score']:.1f}/10")
                            st.markdown("**Factors:**")
                            for factor in sig['factors']:
                                st.markdown(f"- {factor}")
            
            st.markdown("---")
            
            # Full table
            st.markdown("### 📋 All Signals")
            
            df_data = []
            for sig in signals:
                df_data.append({
                    'Ticker': sig['ticker'],
                    'Score': f"{sig['score']:.1f}",
                    'Factors': ', '.join([f.split(' ', 1)[1] if ' ' in f else f for f in sig['factors']])
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download option
            st.download_button(
                label="📥 Download Results (CSV)",
                data=df.to_csv(index=False),
                file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Configure your settings in the sidebar and click 'Run Analysis' to start")
        
        # Show instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Get a FRED API Key** (free):
               - Visit https://fred.stlouisfed.org
               - Create an account
               - Go to "My Account" → "API Keys"
               - Copy your key and paste it in the sidebar
            
            2. **Add Your Watchlist**:
               - Enter stock tickers separated by commas
               - Example: AAPL,MSFT,GOOGL
            
            3. **Run Analysis**:
               - Click the "Run Analysis" button
               - Wait for data collection (1-2 minutes)
               - Review your signals!
            
            **Signal Factors:**
            - 🟢 RSI Oversold (bullish)
            - 🏛️ Congressional Buys (insider activity)
            - 📈 Bullish Sentiment (social media)
            - 📊 Volume Spike (increased interest)
            - 🌤️ Low Volatility (stable market)
            """)
        
        with st.expander("⚠️ Disclaimer"):
            st.markdown("""
            This tool is for educational purposes only. 
            
            - Not financial advice
            - Past performance doesn't guarantee future results
            - Always do your own research
            - Never invest more than you can afford to lose
            """)


if __name__ == "__main__":
    main()
