import requests
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class BTCMarkPriceCompleteBayesianTest:
    def __init__(self, symbol="XBTUSDTM"):
        self.symbol = symbol
        self.api_base = "https://api-futures.kucoin.com"
        
    def fetch_btc_historical_data(self, days=180):
        """Fetch complete BTC mark price data from KuCoin Futures"""
        print(f"🔄 Fetching {days} days of {self.symbol} 1-minute mark price data...")
        print(f"📊 This will collect ~{days * 24 * 60:,} data points")
        
        all_data = []
        end_time = int(time.time() * 1000)  # milliseconds
        total_minutes = days * 24 * 60
        max_per_request = 200  # KuCoin futures limit
        total_requests = (total_minutes // max_per_request) + 1
        
        print(f"🔢 Required API calls: {total_requests}")
        print(f"⏱️ Estimated time: {total_requests * 0.5 / 60:.1f} minutes")
        
        successful_requests = 0
        failed_requests = 0
        
        for i in range(total_requests):
            chunk_end = end_time - (i * max_per_request * 60 * 1000)
            chunk_start = chunk_end - (max_per_request * 60 * 1000)
            
            target_start = end_time - (days * 24 * 60 * 60 * 1000)
            if chunk_start < target_start:
                chunk_start = target_start
            
            try:
                url = f"{self.api_base}/api/v1/kline/query"
                params = {
                    'symbol': self.symbol,
                    'granularity': 1,      # 1-minute intervals
                    'from': chunk_start,   # milliseconds
                    'to': chunk_end        # milliseconds
                }
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if data['code'] == '200000' and data['data']:
                    chunk_data = []
                    for candle in data['data']:
                        chunk_data.append({
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'high': float(candle[2]), 
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        })
                    
                    if chunk_data:
                        chunk_df = pd.DataFrame(chunk_data)
                        all_data.append(chunk_df)
                        successful_requests += 1
                        
                        if i % 20 == 0:
                            print(f"✅ Progress: {i+1}/{total_requests} ({(i+1)/total_requests*100:.1f}%)")
                
                else:
                    print(f"⚠️ Chunk {i+1} failed: {data}")
                    failed_requests += 1
                
                time.sleep(0.5)  # Rate limiting
                
                if chunk_start <= target_start:
                    print(f"🎯 Reached target start time, stopping at chunk {i+1}")
                    break
                    
            except Exception as e:
                print(f"❌ Error fetching chunk {i+1}: {e}")
                failed_requests += 1
                time.sleep(2)
                continue
        
        print(f"📈 API Call Summary: {successful_requests} successful, {failed_requests} failed")
        
        if not all_data:
            raise Exception("❌ No data fetched from KuCoin!")
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        completion_rate = len(df) / total_minutes * 100
        print(f"🎯 Total BTC data points: {len(df):,}")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"📊 Data completeness: {completion_rate:.2f}%")
        
        return df
    
    def calculate_all_btc_indicators(self, df):
        """Calculate ALL 80+ technical indicators for BTC"""
        print("🔧 Calculating ALL 80+ technical indicators for BTC...")
        
        # Moving Averages (comprehensive set)
        for period in [3, 5, 10, 20, 50]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # MACD (multiple timeframes)
        df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=26, window_fast=12)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12)
        
        # Fast MACD
        df['macd_fast'] = ta.trend.macd(df['close'], window_slow=12, window_fast=6)
        df['macd_fast_signal'] = ta.trend.macd_signal(df['close'], window_slow=12, window_fast=6)
        
        # Ultra-fast MACD
        df['macd_ultra'] = ta.trend.macd(df['close'], window_slow=8, window_fast=4)
        df['macd_ultra_signal'] = ta.trend.macd_signal(df['close'], window_slow=8, window_fast=4)
        
        # RSI with multiple periods
        for period in [3, 5, 7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # Bollinger Bands (multiple settings)
        for window, dev in [(20, 2), (10, 2), (5, 1.5)]:
            bb = ta.volatility.BollingerBands(df['close'], window=window, window_dev=dev)
            df[f'bb_upper_{window}'] = bb.bollinger_hband()
            df[f'bb_middle_{window}'] = bb.bollinger_mavg()
            df[f'bb_lower_{window}'] = bb.bollinger_lband()
            if window == 20:
                df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
        
        # Stochastic Oscillator (multiple speeds)
        for window in [3, 5, 14]:
            df[f'stoch_k_{window}'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=window)
            df[f'stoch_d_{window}'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=window)
        
        # Williams %R (multiple periods)
        for period in [3, 7, 14]:
            df[f'williams_r_{period}'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=period)
        
        # Average True Range
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_7'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
        df['atr_normalized'] = df['atr_14'] / df['close']
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
        
        # Price momentum indicators
        for period in [2, 3, 5, 10, 15, 30]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Volatility indicators
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        df['volatility_spike'] = df['volatility_5'] / df['volatility_5'].rolling(window=20).mean()
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        print("✅ All 80+ technical indicators calculated")
        return df
    
    def create_btc_features(self, df):
        """Create comprehensive BTC feature vectors"""
        print("🎯 Creating BTC feature vectors (no look-ahead bias)...")
        
        # Raw technical indicators
        base_features = [
            # Moving averages
            'sma_3', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_3', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
            
            # MACD variants
            'macd', 'macd_signal', 'macd_histogram',
            'macd_fast', 'macd_fast_signal',
            'macd_ultra', 'macd_ultra_signal',
            
            # RSI variants
            'rsi_3', 'rsi_5', 'rsi_7', 'rsi_14', 'rsi_21',
            
            # Bollinger Bands
            'bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_width_20',
            'bb_upper_10', 'bb_middle_10', 'bb_lower_10',
            'bb_upper_5', 'bb_middle_5', 'bb_lower_5',
            
            # Stochastic
            'stoch_k_3', 'stoch_d_3', 'stoch_k_5', 'stoch_d_5', 
            'stoch_k_14', 'stoch_d_14',
            
            # Williams %R
            'williams_r_3', 'williams_r_7', 'williams_r_14',
            
            # ATR
            'atr_14', 'atr_7', 'atr_normalized',
            
            # Volume
            'volume_sma_20', 'volume_ratio', 'obv', 'obv_sma_20',
            
            # Momentum
            'momentum_2', 'momentum_3', 'momentum_5', 'momentum_10', 
            'momentum_15', 'momentum_30',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20', 
            'volatility_ratio', 'volatility_spike',
            
            # Trend
            'adx'
        ]
        
        # Create binary features
        binary_features = []
        
        # Price position relative to moving averages
        for period in [3, 5, 10, 20, 50]:
            feat_name = f'price_above_sma_{period}'
            binary_features.append(feat_name)
            df[feat_name] = (df['open'] > df[f'sma_{period}']).astype(int)
            
            feat_name = f'price_above_ema_{period}'
            binary_features.append(feat_name)
            df[feat_name] = (df['open'] > df[f'ema_{period}']).astype(int)
        
        # MACD signals
        macd_features = ['macd_bullish', 'macd_above_zero', 'macd_hist_positive',
                        'macd_fast_bullish', 'macd_ultra_bullish']
        binary_features.extend(macd_features)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)
        df['macd_hist_positive'] = (df['macd_histogram'] > 0).astype(int)
        df['macd_fast_bullish'] = (df['macd_fast'] > df['macd_fast_signal']).astype(int)
        df['macd_ultra_bullish'] = (df['macd_ultra'] > df['macd_ultra_signal']).astype(int)
        
        # RSI signals
        for period in [3, 5, 7, 14, 21]:
            feat_names = [f'rsi_{period}_oversold', f'rsi_{period}_overbought', f'rsi_{period}_bullish']
            binary_features.extend(feat_names)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_bullish'] = (df[f'rsi_{period}'] > 50).astype(int)
        
        # Stochastic signals
        for window in [3, 5, 14]:
            feat_names = [f'stoch_bullish_{window}', f'stoch_oversold_{window}', f'stoch_overbought_{window}']
            binary_features.extend(feat_names)
            df[f'stoch_bullish_{window}'] = (df[f'stoch_k_{window}'] > df[f'stoch_d_{window}']).astype(int)
            df[f'stoch_oversold_{window}'] = (df[f'stoch_k_{window}'] < 20).astype(int)
            df[f'stoch_overbought_{window}'] = (df[f'stoch_k_{window}'] > 80).astype(int)
        
        # Williams %R signals
        for period in [3, 7, 14]:
            feat_names = [f'williams_oversold_{period}', f'williams_overbought_{period}']
            binary_features.extend(feat_names)
            df[f'williams_oversold_{period}'] = (df[f'williams_r_{period}'] < -80).astype(int)
            df[f'williams_overbought_{period}'] = (df[f'williams_r_{period}'] > -20).astype(int)
        
        # Volume signals
        volume_features = ['high_volume', 'volume_breakout']
        binary_features.extend(volume_features)
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        df['volume_breakout'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # Momentum signals
        for period in [2, 3, 5, 10, 15, 30]:
            feat_name = f'momentum_{period}_positive'
            binary_features.append(feat_name)
            df[feat_name] = (df[f'momentum_{period}'] > 0).astype(int)
        
        # Volatility signals
        vol_features = ['high_volatility', 'volatility_expansion', 'volatility_spike_high']
        binary_features.extend(vol_features)
        df['high_volatility'] = (df['volatility_ratio'] > 1.2).astype(int)
        df['volatility_expansion'] = (df['volatility_ratio'] > 1.5).astype(int)
        df['volatility_spike_high'] = (df['volatility_spike'] > 2.0).astype(int)
        
        # Combine all features
        all_features = base_features + binary_features
        
        # Target variable (CORRECTED - no look-ahead bias)
        df['target'] = (df['close'] > df['open']).astype(int)
        
        print(f"✅ Created {len(all_features)} features total")
        return df, all_features
    
    def create_lagged_features(self, df, features):
        """Create lagged features to avoid look-ahead bias"""
        print("🔄 Creating lagged features (using previous minute data only)...")
        
        # Shift all features by 1 to use only previous minute's data
        lagged_features = []
        for feature in features:
            lagged_name = f'{feature}_lag1'
            df[lagged_name] = df[feature].shift(1)
            lagged_features.append(lagged_name)
        
        # Remove rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ Created {len(lagged_features)} lagged features")
        print(f"📊 Clean data points: {len(df):,}")
        
        return df, lagged_features
    
    def backtest_rolling_windows(self, df, lagged_features):
        """Backtest with rolling windows (no look-ahead bias)"""
        print("🔄 Starting rolling window backtesting...")
        
        # Training window sizes
        training_windows = {
            "1 Month": 30 * 24 * 60,    # 43,200 minutes
            "2 Months": 60 * 24 * 60,   # 86,400 minutes
            "3 Months": 90 * 24 * 60,   # 129,600 minutes
            "6 Months": 180 * 24 * 60   # 259,200 minutes
        }
        
        test_period = 7 * 24 * 60  # 1 week = 10,080 minutes
        results = {}
        
        for window_name, window_size in training_windows.items():
            print(f"\n🔄 Testing {window_name} training window...")
            
            # Initialize tracking variables
            balance = 1.0
            position = None  # None, 'long', 'short'
            trades = []
            predictions = []
            actual_targets = []
            confidences = []
            
            # Test on last week
            test_start_idx = len(df) - test_period
            
            for i in range(test_start_idx, len(df)):
                # Define training window (no future data)
                train_start_idx = max(0, i - window_size)
                train_end_idx = i
                
                # Skip if insufficient training data
                if train_end_idx - train_start_idx < 1000:
                    continue
                
                # Prepare training data
                train_data = df.iloc[train_start_idx:train_end_idx]
                X_train = train_data[lagged_features].values
                y_train = train_data['target'].values
                
                # Train model
                model = GaussianNB()
                model.fit(X_train, y_train)
                
                # Make prediction using ONLY previous minute's data
                current_row = df.iloc[i]
                X_current = current_row[lagged_features].values.reshape(1, -1)
                
                prediction = model.predict(X_current)[0]
                prediction_prob = model.predict_proba(X_current)[0]
                confidence = prediction_prob.max()
                
                # Store for evaluation
                predictions.append(prediction)
                actual_targets.append(current_row['target'])
                confidences.append(confidence)
                
                # Trading logic (only trade on high confidence)
                if confidence > 0.54:  # Confidence threshold
                    new_direction = 'long' if prediction == 1 else 'short'
                    
                    # Open position if none exists
                    if position is None:
                        position = new_direction
                        trades.append({
                            'timestamp': current_row['timestamp'],
                            'action': 'open',
                            'direction': position,
                            'price': current_row['open'],
                            'balance_before': balance
                        })
                    
                    # Close and reverse if direction changes
                    elif position != new_direction:
                        trades.append({
                            'timestamp': current_row['timestamp'],
                            'action': 'close',
                            'direction': position,
                            'price': current_row['open'],
                            'balance_before': balance
                        })
                        
                        position = new_direction
                        trades.append({
                            'timestamp': current_row['timestamp'],
                            'action': 'open',
                            'direction': position,
                            'price': current_row['open'],
                            'balance_before': balance
                        })
                
                # Calculate P&L for current position
                if position:
                    current_return = 0
                    if position == 'long':
                        current_return = (current_row['close'] - current_row['open']) / current_row['open']
                    else:
                        current_return = (current_row['open'] - current_row['close']) / current_row['open']
                    
                    balance *= (1 + current_return * 0.1)  # 10% position sizing
            
            # Calculate metrics
            if len(predictions) > 0:
                accuracy = accuracy_score(actual_targets, predictions)
                win_rate = np.mean(np.array(predictions) == np.array(actual_targets))
                avg_confidence = np.mean(confidences)
                high_conf_rate = np.mean(np.array(confidences) > 0.54)
                
                results[window_name] = {
                    'accuracy': accuracy,
                    'win_rate': win_rate,
                    'final_balance': balance,
                    'total_trades': len(trades),
                    'avg_confidence': avg_confidence,
                    'high_confidence_rate': high_conf_rate,
                    'profit_percentage': (balance - 1.0) * 100
                }
                
                print(f"✅ {window_name} Results:")
                print(f"   📊 Accuracy: {accuracy:.4f}")
                print(f"   🎯 Win Rate: {win_rate:.2%}")
                print(f"   💰 Final Balance: ${balance:.6f}")
                print(f"   📈 Profit: {(balance - 1.0) * 100:+.4f}%")
                print(f"   🔄 Total Trades: {len(trades)}")
                print(f"   🎯 Avg Confidence: {avg_confidence:.4f}")
                print(f"   📊 High Conf Rate: {high_conf_rate:.2%}")
        
        return results
    
    def run_complete_btc_test(self):
        """Run the complete BTC backtesting system"""
        print("🚀 Starting Complete BTC Mark Price Backtesting System")
        print("📋 Features:")
        print("   • No look-ahead bias")
        print("   • Rolling window training")
        print("   • Previous minute data only")
        print("   • 80+ technical indicators")
        print("   • Real trading simulation")
        print("-" * 60)
        
        # Step 1: Fetch data
        btc_df = self.fetch_btc_historical_data(days=180)
        
        # Step 2: Calculate indicators
        btc_df = self.calculate_all_btc_indicators(btc_df)
        
        # Step 3: Create features
        btc_df, features = self.create_btc_features(btc_df)
        
        # Step 4: Create lagged features (no look-ahead bias)
        btc_df, lagged_features = self.create_lagged_features(btc_df, features)
        
        # Step 5: Backtest with rolling windows
        results = self.backtest_rolling_windows(btc_df, lagged_features)
        
        # Step 6: Display final summary
        print("\n" + "="*60)
        print("🎯 FINAL BTC BACKTESTING RESULTS")
        print("="*60)
        
        for window_name, metrics in results.items():
            print(f"\n📊 {window_name}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Win Rate: {metrics['win_rate']:.2%}")
            print(f"   Profit: {metrics['profit_percentage']:+.4f}%")
            print(f"   Final Balance: ${metrics['final_balance']:.6f}")
            print(f"   Total Trades: {metrics['total_trades']}")
            print(f"   High Confidence Rate: {metrics['high_confidence_rate']:.2%}")
        
        # Find best performing window
        best_window = max(results.keys(), key=lambda k: results[k]['final_balance'])
        best_metrics = results[best_window]
        
        print(f"\n🏆 BEST PERFORMING WINDOW: {best_window}")
        print(f"   📈 Profit: {best_metrics['profit_percentage']:+.4f}%")
        print(f"   🎯 Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   💰 Final Balance: ${best_metrics['final_balance']:.6f}")
        
        return results

# Run the complete system
if __name__ == "__main__":
    print("🚀 BTC Complete Backtesting System")
    print("⚠️  This will take 10-15 minutes to fetch all data")
    print("📊 Testing periods: 1, 2, 3, 6 months")
    print("🎯 No look-ahead bias - uses only previous minute data")
    print("-" * 60)
    
    tester = BTCMarkPriceCompleteBayesianTest()
    results = tester.run_complete_btc_test()
    
    print("\n✅ Complete BTC backtesting finished!")
    print("🎯 Results show realistic live trading performance expectations")
