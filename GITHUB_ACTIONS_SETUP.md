# GitHub Actions Daily Price Update Setup

This guide explains how to set up automatic daily price updates for your Pokemon cards database using GitHub Actions.

## 🚀 What This Does

- **Runs daily at 12:00 PM UTC** (noon)
- **Updates prices** for all Pokemon cards in your database
- **Commits changes** back to your repository
- **Stores database** as GitHub artifact for 30 days
- **Provides detailed logs** of the update process

## 📋 Prerequisites

1. **Database exists**: Make sure `poke_backend/pokemon_cards.db` exists
2. **API key**: Your Pokemon TCG API key (optional, uses default if not set)

## 🔧 Setup Instructions

### 1. Push Your Database

First, commit and push your current database to GitHub:

```bash
git add poke_backend/pokemon_cards.db
git commit -m "Add initial Pokemon cards database"
git push origin main
```

### 2. Set Up API Key (Optional)

If you want to use your own API key instead of the default one:

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `POKEMON_TCG_API_KEY`
5. Value: Your API key from [Pokemon TCG Developer Portal](https://dev.pokemontcg.io/)

### 3. Enable GitHub Actions

The workflow will automatically start running once you push the files:

```bash
git add .github/workflows/daily_price_update.yml
git add poke_backend/pokeExp_github_actions.py
git commit -m "Add GitHub Actions daily price update workflow"
git push origin main
```

## 📊 How It Works

### Daily Schedule
- **Time**: 12:00 PM UTC (noon) every day
- **Duration**: ~10-15 minutes
- **Runs on**: Ubuntu latest

### Process Flow
1. **Checkout** your repository
2. **Download** previous database artifact (if exists)
3. **Install** Python dependencies
4. **Run** price update script
5. **Upload** updated database as artifact
6. **Commit** and push changes to repository

### Manual Trigger
You can also run the workflow manually:
1. Go to **Actions** tab in your repository
2. Click **Daily Pokemon Card Price Update**
3. Click **Run workflow**

## 📈 Monitoring

### View Logs
1. Go to **Actions** tab
2. Click on the latest workflow run
3. Click on **update-prices** job
4. View detailed logs for each step

### Check Results
- **Database**: Updated `poke_backend/pokemon_cards.db` in your repository
- **Artifacts**: Download from Actions tab (30-day retention)
- **Commits**: See daily commits with price updates

## 🔍 Troubleshooting

### Common Issues

**Database not found**
```
❌ Database file not found! Please ensure pokemon_cards.db exists.
```
**Solution**: Make sure you've committed the database file to your repository.

**API rate limiting**
```
API error on page X: Rate limit exceeded
```
**Solution**: The script automatically retries with delays. This is normal.

**Permission denied**
```
Error: fatal: could not read Username for 'https://github.com'
```
**Solution**: Make sure the repository has write permissions for GitHub Actions.

### Debug Mode

To test locally before pushing:

```bash
cd poke_backend
python pokeExp_github_actions.py
```

## 📝 Files Created

- `.github/workflows/daily_price_update.yml` - GitHub Actions workflow
- `poke_backend/pokeExp_github_actions.py` - Optimized price update script
- `GITHUB_ACTIONS_SETUP.md` - This setup guide

## 🎯 Benefits

✅ **Automatic**: No manual intervention needed  
✅ **Reliable**: GitHub's infrastructure handles execution  
✅ **Trackable**: Full logs and commit history  
✅ **Scalable**: Can handle thousands of cards  
✅ **Secure**: API keys stored as secrets  
✅ **Free**: GitHub Actions provides 2000 minutes/month free  

## 📅 Next Steps

1. **Monitor** the first few runs to ensure everything works
2. **Check** the database updates in your repository
3. **Verify** price data is being collected correctly
4. **Customize** the schedule if needed (edit the cron expression)

The workflow will now run automatically every day at noon UTC! 🎉 