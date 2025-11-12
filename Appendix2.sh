cd ~/competition-bot && cat > fix_trading_pairs.py << 'FIXEOF'
#!/usr/bin/env python3
"""
CRITICAL FIX: Force trading pairs to load during bot initialization
"""

# Read the current bot code
with open('ultimate_final_bot.py', 'r') as f:
    content = f.read()

# Find the __init__ method and fix the trading pairs initialization
old_init_code = '''        self.trading_pairs=[]
        self.price_history={}
        self.volume_history={}
        self.positions={}'''

new_init_code = '''        # CRITICAL FIX: Load trading pairs immediately
        self.trading_pairs=self.api.get_all_pairs()
        self.price_history={}
        self.volume_history={}
        self.positions={}'''

if old_init_code in content:
    content = content.replace(old_init_code, new_init_code)
    print("✅ FIXED: Trading pairs now load during initialization")
else:
    print("❌ Could not find exact code pattern - checking alternatives...")
    
    # Alternative fix - look for trading_pairs initialization
    if "self.trading_pairs=[]" in content:
        content = content.replace("self.trading_pairs=[]", "self.trading_pairs=self.api.get_all_pairs()  # CRITICAL FIX")
        print("✅ FIXED: Trading pairs initialization replaced")
    else:
        print("❌ Manual code inspection needed")

# Write the fixed code
with open('ultimate_final_bot.py', 'w') as f:
    f.write(content)

print("🚀 Bot code updated - trading pairs should load properly now")
FIXEOF

cd ~/competition-bot && python3 fix_trading_pairs.py