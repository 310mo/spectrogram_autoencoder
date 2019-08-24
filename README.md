# spectrogram_autoencoder

やってることの詳しい内容は[ここ](http://kanoeuma-310mo.hatenablog.jp/entry/2019/08/25/004846)に書いてある。

analyzeがスペクトログラムの出力とか、grifin-lim法の実装とかが置いてある方で、autoencoderの方がオートエンコーダの実装が置いてある方。

ディレクトリ作ってそこにデータを置いて
```
python autoencoder.py
```
とすれば学習できる。学習データのディレクトリ名はspect-16000になってて（prepro.pyから変更できる）
その中にtrainディレクトリとvalディレクトリを配置することを想定している。ここらへんもprepro.pyに記述してある。

net.ckptが学習済みの重み。net_check.pyがそれを使ってテストするコードになっている。

encoded.pyはnet_check.pyとほとんど同じなんだけど、こっちは中間の特徴量を出力する用。
