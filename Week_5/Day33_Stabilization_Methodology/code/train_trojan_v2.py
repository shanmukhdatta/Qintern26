import sys, time
sys.path.insert(0, '.')
import config
import data_prep
import train_qgan
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.FileHandler(f'{config.LOG_DIR}/trojan_v2_train.log', mode='a'),
                               logging.StreamHandler(sys.stdout)])

df = data_prep.load_raw()
feature_cols = data_prep.get_feature_columns(df)
by_class = data_prep.split_by_class(df, feature_cols)
X_real = by_class['Trojan']
rep = data_prep.ClassRepresentation()
X_norm = rep.fit_transform(X_real)
print(f"Trojan n_real={len(X_real)}, N_VARIATIONAL_LAYERS={config.N_VARIATIONAL_LAYERS}, DISC_DROPOUT_P={config.DISC_DROPOUT_P}")
t0 = time.time()
result = train_qgan.train_one_class_safe(X_norm, 'Trojan', verbose=True)
print(f"chunk done, {time.time()-t0:.1f}s this call")
