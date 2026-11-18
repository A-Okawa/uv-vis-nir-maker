# UV-Vis-NIR Maker

紫外・可視・近赤外分光（UV-Vis-NIR）のスペクトルを論文用の図にする Streamlit アプリです。XRD Maker・Raman Maker と同じ構成・同じ体裁で、
色の組み合わせ（`palettes.py`）、文字の書式（`rich_text.py`）、図の体裁と出力（`figure_style.py`）、フォント（`fonts/`）、色の名前（`colors.csv`）は
XRD Maker と同じファイルを共有しています。片方を直したら、もう片方にもコピーしてください。

## できること

- **読み込み**：JASCO（V-650/V-670/V-770 の Spectra Manager テキスト・CSV）、Shimadzu UVProbe（テキスト・CSV）、Agilent Cary（CSV、試料ごとに波長列と値列が並ぶ形式）、
  Hitachi UH4150/U-4100 と PerkinElmer Lambda のテキスト・CSV、見出しつき／なしの一般的な表（カンマ・セミコロン・タブ・空白区切り、小数点カンマ可）、
  JCAMP-DX（`.jdx`/`.dx`、AFFN と SQZ/DIF/DUP 圧縮）、Excel（`.xlsx`、全シート）。1 ファイルに複数の列・ブロックがあれば別々のスペクトルになります。
- **縦軸の量**：測定値のまま、透過率 T、吸光度 A = −log₁₀T、反射率 R、Kubelka–Munk 関数 F(R) = (1−R)²/2R、Tauc プロット (αhν)^{1/n} vs hν。
  T ↔ A、R ↔ F(R) は互いに換算します（透過率のデータを反射率にはできません）。値の種類は見出し（`%T`、`Abs.`、`%R` など）から読み、見出しがないときは
  値の範囲から判断します（サイドバーの「ヘッダーに記載がないときの読み方」で指定もできます）。
- **横軸の単位**：nm・µm・eV・cm⁻¹。範囲は nm で指定し、表示だけを換えます。
- **Tauc プロット**：直接許容（n = 1/2）・間接許容（n = 2）・直接禁制・間接禁制。α は吸光度（透過データ）または F(R)（反射データ）に比例するとして描き、
  指定した範囲の直線を当てはめて Y = 0 へ延長した交点を E_g として図と表に出します。
- **光学指標**：T_lum（CIE 1931 の視感度 ȳ(λ)、380〜780 nm）、T_sol・T_vis・T_nir・T_uv（AM1.5G、ASTM G173-03、300〜2500・380〜780・780〜2500・300〜380 nm）。
  反射率のデータでは同じ重みで R の平均（顔料の NIR 日射反射率）。表は CSV で保存できます。重みのデータは `data/`（CIE 1931 2° ȳ は 1 nm 刻み、ASTM G173-03 は NREL の表を pvlib 経由で複製）。
- **図**：正規化（最大値・指定範囲・最小最大・指定波長の値・面積）、積み重ね／重ね描き（Y位置と倍率）、線種、凡例の位置、縦軸の範囲と目盛り、
  極大・極小の位置、注記の破線と帯（UV / Vis / NIR）、拡大図、サンプルラベル、Savitzky–Golay 平滑化。
- **出力**：PDF / SVG（ベクトル）、PNG / TIFF（300 / 600 DPI）。作業ファイル（.json）と作図スタイル（.json、XRD Maker・Raman Maker のスタイルも読めます）。

## PC 版

UV-Vis-NIR Maker を PC にダウンロードして、ブラウザーでローカル実行できます。
**初回セットアップには Python と依存パッケージが必要です。通常の ZIP は Python を含みません。**
一度セットアップすれば、起動・データ読込・描画・プロジェクトの保存／再開はインターネット接続なしで利用できます。

### Windows

1. Python 3.12 をインストールします。Python のインストーラーでは「Add Python to PATH」を選択してください。
2. `UV-Vis-NIR-Maker-PC.zip` をすべて展開します。ZIP 内から直接実行しないでください。
3. `setup_windows.bat` をダブルクリックします。初回はパッケージのダウンロードに数分かかる場合があります。
4. セットアップ完了後、`launch_windows.bat` をダブルクリックします。ブラウザーが自動で開きます。
5. 終了するときは、起動時に開いたコマンドプロンプトで **Ctrl+C** を押してください。

### macOS

1. Python 3.12 をインストールします（Python 3.10 以降に対応、検証環境は Python 3.12）。
2. ZIP をすべて展開し、`setup_mac.command` を開きます。
3. セットアップ完了後、`launch_mac.command` を開きます。ブラウザーが自動で開きます。
4. 終了するときは、起動時に開いたターミナルで **Ctrl+C** を押してください。

`.command` を開けない場合は、ターミナルで展開先へ移動し、以下を実行できます。

```sh
bash setup_mac.command
bash launch_mac.command
```

### Linux

```sh
bash setup_linux.sh
bash launch_linux.sh
```

### 保存先

| OS | 既定の保存先 |
| --- | --- |
| macOS | `~/Library/Application Support/UV-Vis-NIR Maker/` |
| Windows | `%LOCALAPPDATA%\UV-Vis-NIR Maker\` |
| Linux | `~/.local/share/uvvisnir-maker/`（`XDG_DATA_HOME` 設定時はそちらを使用） |

ローカル自動保存はこのフォルダー内の `autosave.json`、起動ログは `launcher.log` に保存します。
保存先を分けたい場合は `launch_linux.sh --data-dir /任意の保存先` のように指定できます（Mac の launch ファイルや Windows の `.bat` も同じ引数を受け付けます）。
PC 版は `127.0.0.1` のみに接続し、起動時に認証情報を設定する必要はありません。

### 初回セットアップもオフラインにする

対象 PC と同じ OS・CPU・Python のオンライン環境で wheel を取得し、展開したフォルダーに `wheelhouse` として置くと、setup はそこからだけインストールします。

```sh
python -m pip download --only-binary=:all: -r requirements.txt --dest wheelhouse
python setup_local.py --offline --wheelhouse ./wheelhouse
```

### 開発者向け：配布 ZIP の作成

```sh
python build_bundle.py --output dist/UV-Vis-NIR-Maker-PC.zip
python build_bundle.py --wheelhouse ./wheelhouse --output dist/UV-Vis-NIR-Maker-PC-with-wheels.zip
```

## Web 版（Streamlit Community Cloud）

XRD Maker と同じ手順です。GitHub のリポジトリを Streamlit Community Cloud に接続し、`app.py` を指定します。
Secrets に `password = "…"` を設定するとログイン画面が出ます（PC 版では不要）。`requirements.txt` のパッケージだけで動きます。

## 論文用の作図（XRD Maker・Raman Maker と同じ）

- 新しい作業は「グラフ描画appと同じ体裁 · 88 × 70 mm」（Arial 7 pt、目盛り・凡例 6.5 pt、軸 0.5 pt、曲線 0.9 pt、目盛り 2.5 pt で外向き、上と右の枠なし）で始まります。
  投稿先の指定に合わせて「図サイズ・出力」と「線・フォント・余白」で調整してください。
- 色の組み合わせ：論文用（青・朱・緑…）、Okabe–Ito、青・朱・緑・灰の濃淡、viridis、matplotlib 標準。濃淡の一番薄い色も白地とのコントラスト 3:1 以上。
  「色を付ける範囲（表示順）」で、基板や参照試料を範囲外の色（初期値は黒）にできます。`colors.csv` の名前付きの色（750 °C など）も選べます。
- グループごとの濃淡：「グループの大きさ」に 3 を入れると、表示順に 3 本ずつを 1 グループにして、1 グループ目は青の濃淡、2 グループ目は黄の濃淡、3 グループ目は朱の濃淡…になります
  （同じ試料をふるいの粒径違いで測ったときなど）。色の順は「グループごとの色」で変えられます。黄は白地で見えるよう、金色〜黄土色の範囲を使います。
- 凡例：「図に収める」（初期値はオン）で、凡例が図や軸の枠からはみ出すときに自動で収めます。「図の外（右）」では軸を縮めて場所を空け、図の中では文字を必要なだけ小さくします。「凡例の列数」も指定できます。
- 文字の書式：`<i>…</i>`、`_{…}`、`^{…}`。入力欄で文字を選んで Ctrl+I / Ctrl+B / Ctrl+, / Ctrl+.（Mac は ⌘ でも可）。例：`Wavelength, <i>λ</i> (nm)`、`VO_{2}`、`(<i>αhν</i>)^{2} (a.u.)`。
- 軸のラベルは、単位や縦軸の量を変えると、書き換えていない限り自動で追従します（`Transmittance (%)` → `Absorbance` など）。
- 注記の破線・帯は、図の横軸と同じ単位で 1 行に 1 つ書きます（`550 <i>T</i>_{lum}`、`780-2500 NIR`）。「UV / Vis / NIR の帯を追加」で 3 つの帯が入ります。
- 作図スタイル（.json）は寸法・線・フォント・文字の大きさ・ラベル位置だけを保存します。XRD Maker・Raman Maker のスタイルを開くと、共通の項目だけ取り込みます。

## 測定データの対応形式

| 装置・形式 | 対応内容 |
| --- | --- |
| JASCO `.txt`・`.csv` | Spectra Manager のテキスト／CSV 出力（`TITLE`〜`XYDATA`、`##### Extended Information`）。`XUNITS`・`YUNITS`（%T、ABS、%R …）と測定条件を読みます |
| Shimadzu UVProbe `.txt`・`.csv` | `"試料.spc - RawData"` の見出しと `"Wavelength nm." "Abs."` の列。複数の RawData ブロックは別スペクトル |
| Agilent Cary `.csv` | 1 行目に試料名、2 行目に `Wavelength (nm),%T,…` が繰り返す形式。列の長さが違っても可 |
| Hitachi・PerkinElmer `.txt`・`.csv` | ヘッダー行のあとに `nm %T` などの列見出しと数値 |
| 一般の表 `.txt`・`.csv`・`.dat`・`.asc`・`.xy`・`.tsv`・`.prn` | 1 列目が波長、2 列目以降が値。見出し行があれば列名がスペクトル名、単位（nm・eV・cm⁻¹・µm）と値の種類（%T・Abs・%R …）を見出しから読みます |
| JCAMP-DX `.jdx`・`.dx`・`.jcamp` | `XYDATA=(X++(Y..Y))`（AFFN・SQZ・DIF・DUP）、`XYPOINTS`、`PEAK TABLE`。複合ファイルはブロックごと |
| Excel `.xlsx` | 各シートを上の表として読みます |

バイナリー（`.spc`、`.jws`、`.sp` など）は対象外です。装置ソフトでテキスト／CSV に書き出してください。
波長の単位や値の種類を見出しから判断できない 2 列のデータは、「ヘッダーに記載がないときの読み方」の設定で読みます（初期値は nm、値は範囲から自動判断：0〜100 なら %T、0〜1 なら T、それ以外は吸光度）。

## 保存ファイルについて

「作業を保存・ダウンロード」で `uvvisnir_project.json` を保存します。元のファイル、各スペクトルの表示順・表示／非表示・ラベル・色・線種・オフセット、縦軸の量、図の設定を含みます。
再開するときは「保存した作業を開く」でこの JSON を選びます。PC 版は操作が確定するたびに `autosave.json` を更新し、「前回の自動保存から再開」で続きから作業できます。

## テスト

```sh
python -m pytest tests -q
```

`tests/data/` の合成データ（各形式）で読み込み・換算・図の生成・保存／再開を確認します。Streamlit 1.32 と 1.66 で確認しています。
