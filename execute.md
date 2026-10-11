pip install -r requirements.txt

python test_model.py                              # 1. sanity-check first, works offline
python download_data.py --search "hindi"           # 2. find a Kaggle dataset
python download_data.py --dataset owner/slug        #    download it
python scraper.py wiki --requests 100 --delay 3     # 3. (optional) scrape more Hindi text
python build_vocab.py --min-count 3                 # 4. build the real vocabulary
python test_model.py --vocab                        # 5. verify the real model
streamlit run app.py                                # 6. demo it