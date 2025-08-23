# 🚀 Quick Setup Guide

Get your AI Trading System running in just 5 minutes!

## ⚡ Prerequisites
- Python 3.8+ installed
- Upstox account with API access
- (Optional) Gemini or Azure OpenAI API key

## 📋 Step-by-Step Setup

### 1. Download and Extract
Download the trading system files to your computer.

### 2. Create Configuration File
Create a file named `.env` in the main folder with your API keys:

```env
# REQUIRED: Your Upstox API credentials
UPSTOX_API_KEY=your_actual_upstox_api_key_here
UPSTOX_API_SECRET=your_actual_upstox_secret_here

# OPTIONAL: AI API keys (choose at least one for enhanced features)
GEMINI_API_KEY=your_gemini_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here

# Risk settings (you can keep these as default)
MAX_RISK_PER_TRADE=0.005
DAILY_LOSS_CAP=0.03
MAX_TRADES_PER_DAY=200
```

### 3. Start the System

#### On Windows:
Double-click `start.bat`

#### On Mac/Linux or Advanced Users:
```bash
python start.py
```

### 4. Access Your Dashboard
Open your web browser and go to:
```
http://localhost:8000
```

## 🎯 What Happens Next?

1. **System Check**: The startup script will:
   - Install required packages automatically
   - Validate your API keys
   - Run basic tests
   - Start the trading system

2. **Dashboard**: You'll see:
   - Real-time portfolio overview
   - Trading signals from AI and technical analysis
   - Risk monitoring
   - Order execution interface

3. **Trading**: The system will:
   - Analyze market data continuously
   - Generate trading signals
   - Execute trades (if signals meet criteria)
   - Monitor risk and performance

## ⚠️ Important First Steps

### Start with Paper Trading
1. Set very small position sizes initially
2. Monitor the system for a few days
3. Gradually increase position sizes as you gain confidence

### Monitor the Dashboard
- Check the dashboard regularly during market hours
- Watch for any error messages or alerts
- Review daily performance reports

### Safety Features
- The system has built-in stop-losses
- Daily loss caps prevent large losses
- Risk management has ultimate veto power
- You can manually stop trades anytime

## 🆘 Need Help?

### Common Issues:

**"Python not found"**
- Install Python from python.org
- Make sure it's added to your system PATH

**"API connection failed"**
- Double-check your API keys in the .env file
- Ensure your Upstox API is activated
- Check your internet connection

**"Dashboard won't load"**
- Wait 2-3 minutes for full startup
- Try refreshing your browser
- Check that port 8000 isn't blocked

**"No trading signals"**
- This is normal during low-activity periods
- The system waits for high-confidence signals
- Check that your AI API keys are correctly configured

### Getting Support:
1. Check the full README.md for detailed documentation
2. Look at the logs in the `logs/` folder
3. Report issues with detailed error messages

## 🎉 You're All Set!

Your AI Trading System is now running! The system will:
- ✅ Monitor multiple data sources
- ✅ Generate AI-powered trading signals  
- ✅ Execute trades with strict risk management
- ✅ Provide real-time performance monitoring

**Happy Trading! 📈**

---
*Remember: Start small, monitor closely, and never risk more than you can afford to lose.*