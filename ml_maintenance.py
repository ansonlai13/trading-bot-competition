#!/usr/bin/env python3
from ultimate_final_bot import EnhancedMLPredictor
ml = EnhancedMLPredictor()

print('=== ENSURING CONSISTENT ML TRAINING ===')

# Test and retrain ALL models for consistency
all_pairs = list(ml.models.keys())
print(f'Testing {len(all_pairs)} models for consistency...')

for pair in all_pairs:
    bull = [0.03,0.05,0.08,0.3,0.02,0.01,1.5,0.0001,1.03,0.02,0.04,0.01]
    bear = [-0.03,-0.05,-0.08,0.7,-0.02,-0.01,0.7,0.0003,0.97,0.015,-0.03,-0.01]
    
    bull_pred = ml.predict(pair, bull)
    bear_pred = ml.predict(pair, bear)
    variation = abs(bull_pred - bear_pred)
    
    if variation < 0.2:  # Retrain if weak
        print(f'Retraining {pair} (Var={variation:.3f})...')
        ml.train_enhanced_model(pair)
        
        # Test again
        bull_pred = ml.predict(pair, bull)
        bear_pred = ml.predict(pair, bear)
        variation = abs(bull_pred - bear_pred)
        print(f'  After retraining: Var={variation:.3f}')
    else:
        print(f'✅ {pair}: Var={variation:.3f}')

print('=== CONSISTENCY CHECK COMPLETE ===')
