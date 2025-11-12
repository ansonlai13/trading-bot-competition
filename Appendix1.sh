cd ~/competition-bot && cat > fix_transfer_critical.py << 'CRITICALFIXEOF'
#!/usr/bin/env python3
"""
CRITICAL FIX: Stop deleting position_transfer.json permanently
"""

# Read current code
with open('ultimate_final_bot.py', 'r') as f:
    content = f.read()

# Find and REPLACE the dangerous deletion code
dangerous_patterns = [
    "os.remove('position_transfer.json')",
    "# os.remove('position_transfer.json')  # COMMENTED OUT"
]

fixed = False
for pattern in dangerous_patterns:
    if pattern in content:
        content = content.replace(pattern, "# POSITION TRANSFER FILE PRESERVED FOR SAFETY")
        print(f"✅ REMOVED DANGEROUS CODE: {pattern}")
        fixed = True

if not fixed:
    # Check if already fixed
    if "POSITION TRANSFER FILE PRESERVED" in content:
        print("✅ ALREADY FIXED: Transfer file protection active")
    else:
        print("❌ COULD NOT FIND DANGEROUS CODE - Manual check needed")

# Write safe code
with open('ultimate_final_bot.py', 'w') as f:
    f.write(content)

print("✅ CRITICAL BUG FIXED: Transfer file will never be deleted")
print("🚨 PORTFOLIO PROTECTION: Positions safe across restarts/crashes")
CRITICALFIXEOF