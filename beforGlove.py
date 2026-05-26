import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam      # ★追加：学習スピード調整用
from tensorflow.keras.regularizers import l2      # ★追加：L2正則化（過学習防止）用
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight # 自動計算は今回使わないためコメントアウト

# --- 1. 前処理 ---
max_words = 10000
# 大元のデータ(df)が読み込まれている前提です
df_sample = df.sample(n=100000, random_state=20050126)
max_length = 150 # 時短：200から150へ削減！

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_sample['review_text'])
sequences = tokenizer.texts_to_sequences(df_sample['review_text'])
X = pad_sequences(sequences, maxlen=max_length)
y = df_sample['is_spoiler'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20050126)

# --- ★変更：クラスの重み（ペナルティ）を手動でマイルドに設定 ---
# 自動計算のスパルタ（1.91倍）をやめ、1.5倍に手加減して過剰反応を防ぐ
weight_dict = {0: 1.0, 1: 1.5}
print(f"【ペナルティ設定】 ネタバレなし:{weight_dict[0]} / ネタバレあり:{weight_dict[1]} の手動チューニング")


# --- 2. 時短＆高精度のモデル構築（チューニング反映） ---
model = Sequential()

# ★チューニング1：output_dimを128から64へダイエット（丸暗記防止）
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_length))

# 過学習を即座にブロックするため、強めのDropout(0.4)
model.add(Dropout(0.4))

# 脳みそを32に絞って、複雑な問題に対応しつつキャパオーバーを防ぐ
model.add(Bidirectional(LSTM(32)))

model.add(Dropout(0.4))

# ★チューニング2：L2正則化を追加して、特定の特徴への過剰な依存を防ぐ
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

# ★チューニング3：学習スピードを通常の0.001から0.0005へ落として慎重に下らせる
custom_adam = Adam(learning_rate=0.0005)
model.compile(optimizer=custom_adam, loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. 最強の時短術：Early Stopping ---
# エラーが1回悪化しても、あと2回は様子を見る（粘り強く学習させる）
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# --- 4. 学習スタート ---
print("\nチューニング済みの自力学習モデルで学習を開始します...")
history = model.fit(
    X_train, y_train,
    epochs=10,        
    batch_size=128,   
    validation_split=0.2,
    callbacks=[early_stop], 
    class_weight=weight_dict # ★手動ペナルティを適用！
)

# --- 5. 最終テスト ---
loss, lstm_accuracy = model.evaluate(X_test, y_test)
print(f"\n【最終結果】チューニング済みモデルの正解率: {lstm_accuracy * 100:.2f}%")

# 今回は詳しい成績表も出力して、RecallとPrecisionのバランスを確認します
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print("\n=== AIの詳しい成績表（Classification Report） ===")
print(classification_report(y_test, y_pred, target_names=['安全(0)', 'ネタバレ(1)']))