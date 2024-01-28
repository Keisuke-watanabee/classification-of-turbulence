【本取り組みの目的】<br>
・自分の用意した画像データ（修士論文執筆時使用したもの）で２値分類機械学習モデルを作成。乱気流の発生有無を予測する。<br>
・修士論文執筆時にはCNN - ResNet50のみモデルを作成にとどまったが、自身の学習のため一通りの機械学習モデルを使用し、<br>
正常にモデルを動かすことができるかの確認、ハイパーパラメータのチューニング、各モデルの精度比較を行う。<br>
<br>
【使用データ】<br>
・学習用 乱気流発生時の地点周辺の気象場を表した画像（1500枚） 乱気流未発生時の地点周辺の気象場を表した画像（1500枚）<br>
・テスト用 上記２種類の画像データを乱気流ありとなしで３００枚ずつ用意 （詳細は”【研究概要】★使用データ”に記載）<br>
<br>
【使用モデル】<br>
線形SVM、K近傍法、非線形SVM、ランダムフォレスト、XGBoost、ResNet50<br>
<br>
【研究概要】<br>
★研究テーマ 「深層学習を用いた乱気流発生予測モデルの構築」　<br>
<br>
★研究背景<br>
・乱気流の発生は予測が困難であり、飛行機の事故の５０％以上が乱気流に起因するもので、その年間被害額は100万ドルに及ぶ。 　　<br>
・現在の発生予測は地球大気を格子点状に分割（図１）し、格子点ごとに未来の気象情報を予測、その予測値を用いて乱気流発生有無を予測している。　　<br>
　乱気流発生有無の予測に用いるのが、乱気流発生指標（図２）であり、これは乱気流の発生可能性を数式化するものである。　　<br>
　　<br>
![図1](https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/11f8624f-fc37-400d-a770-d9acebe7f9d5)　　<br>
図１　気象予測の数値計算を行うために地球大気を格子点で覆うイメージ　　<br>
![乱気流発生指標例](https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/a6b73980-1096-4f43-9ca7-990e03518585)　　<br>
図２　乱気流発生指標の数式の１例　　<br>
　　<br>
　　<br>
★現状の問題点　　<br>
・乱気流の発生指標の課題　　<br>
…発生指標がどのくらいの値をとる時に乱気流が発生するのかは厳密には不明である。よって発生予測はあくまでも経験則的なものとなっている。そこで過去の発生事例から人工知能によって統計的に発生閾値を算出する研究が注目されている。　　<br>
・機械学習を応用した先行研究の課題　　<br>
…先行研究ではSVMやランダムフォレスト、XGBoostなどが予測モデルとして採用されていた（深層学習の研究は２０２１年時点ではなし）。どの研究も格子点ごとの気象情報のみを使用した予測にとどまっており、格子点周辺の気象情報は考慮できていない。大気は流体であり、周囲の影響を強く受けながら流れていることから、格子点で流体現象を捉えることは難しいことが言える。　　<br>
　　<br>
★本研究の方針　　<br>
・人工知能を用いて乱気流発生予測を行う。　　<br>
・点ではなく広範囲を捉えた”面”による乱気流発生予測を行う。本研究では画像を用いた発生予測を行う。　　<br>
　　<br>
★使用データ作成　　<br>
○元データ　　<br>
・Pilot Report　　<br>
…パイロットによる乱気流の発生した地点・時間の報告　　<br>
・MSM（メソ気象客観解析データ）　　<br>
…気象庁が作成した気象データ。空間解像度が５　kmで日本域で最高解像度。　　<br>
　　本研究では乱気流発生した地点を中心とした320 × 320 kmの気象データを用意（図３）。　　<br>
　　<br>
○画像作成方法　　<br>
・乱気流発生した地点・しなかった地点周辺の乱気流発生指標強度を表す画像を作成。作成した画像（図４）の色の濃さは発生指標強度を表している。この画像データに「乱気流あり」「乱気流なし」とそれぞれラベル付けを行い、モデルの学習を行う。　　<br>
　　<br>
<img width="257" alt="スクリーンショット 2024-01-23 18 47 09" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/1dd45705-d160-4704-8eb6-87aae8a1bd9b">　　<br>
図３　MSMデータの一部　　（X・Y…座標　　U…東西風　　V…南北風　　U_shear…乱気流発生高度から上下2.5km離れた高度の東西風の風速差　　V＿shear…乱気流発生高度から上下2.5km離れた高度の南北風の風速差）　　<br>
　　<br>
<img width="198" alt="スクリーンショット 2024-01-23 18 49 38" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/406feef7-7639-494f-86a4-b06cb4de9311">　　<br>
図４　作成した画像の一部　　<br>
　　<br>
　　<br>
作成した各モデルの精度は下記の通り　　<br>
　　<br>
<img width="229" alt="線形svm" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/0b41e9b8-7715-4c5e-9067-5c3d3ff5d097">　　<br>
線形SVM　　<br>
　　<br>
<img width="233" alt="k近傍" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/9e045c2a-a444-41af-a5e6-6a8135b31d0c">　　<br>
K-NN　　<br>
　　<br>
<img width="234" alt="非線形svm" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/b3494410-efcd-415c-9c5f-2c44535bd036">　　<br>
非線形SVM　　<br>
　　<br>
<img width="224" alt="ランダムフォレスト" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/068086e3-16ab-4737-8c50-f09f24be542f">　　<br>
ランダムフォレスト　　<br>
　　<br>
<img width="222" alt="XGBoost" src="https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/2e1a2c20-9bee-4436-be4d-4084ac59994d">　　<br>
XGBoost　　<br>
　　<br>
![cnn](https://github.com/Keisuke-watanabee/classification-of-turbulence/assets/154974337/6aa79af6-f1e4-4f96-a738-ea401829abc5)　　<br>
CNN-ResNet50 (精度85~90%)　　<br>


