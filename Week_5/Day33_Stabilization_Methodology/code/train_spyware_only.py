import sys, time
sys.path.insert(0, '.')
import config
import data_prep
import train_qgan
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.FileHandler('/home/claude/qgan_work/logs/spyware_train.log', mode='a'),
                               logging.StreamHandler(sys.stdout)])

df = data_prep.load_raw()
feature_cols = data_prep.get_feature_columns(df)
by_class = data_prep.split_by_class(df, feature_cols)
X_real = by_class['Spyware']
rep = data_prep.ClassRepresentation()
X_norm = rep.fit_transform(X_real)
print(f"Spyware n_real={len(X_real)}")
t0 = time.time()
result = train_qgan.train_one_class_safe(X_norm, 'Spyware', verbose=True)
print(f"chunk done, {time.time()-t0:.1f}s this call")
