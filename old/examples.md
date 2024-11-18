```bash
python cleanup.py --input raw_input.csv --output clean_dedup_single_step.csv --remove-emojis --deduplicate --dedup-field place_id --clean-line-terminators --remove-control-characters --normalize-unicode --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews


python cleanup.py --input raw_input.csv --output clean_no_dedup.csv --remove-emojis --clean-line-terminators --remove-control-characters --normalize-unicode --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews

python cleanup.py --input clean_no_dedup.csv --output clean_dedup.csv --deduplicate --dedup-field place_id


python cleanup.py --input raw_input.csv --output dedup_no_clean.csv --deduplicate --dedup-field place_id

python cleanup.py --input dedup_no_clean.csv --output dedup_clean.csv --remove-emojis --clean-line-terminators --remove-control-characters --normalize-unicode --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews




python cleanup3.py --input all-task-318.csv --output clean_dedup_single_step_3.csv --deduplicate --dedup-field place_id --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews --clean-line-terminators --remove-control-characters --remove-emojis --normalize-unicode --n-workers 8 --threads-per-worker 1 --memory-limit 4GB


python cleanup3.py --input all-task-318.csv --output clean_no_dedup_3.csv --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews --clean-line-terminators --remove-control-characters --remove-emojis --normalize-unicode --n-workers 8 --threads-per-worker 1 --memory-limit 4GB

python cleanup3.py --input clean_no_dedup_3.csv --output clean_dedup_3.csv --deduplicate --dedup-field place_id --n-workers 8 --threads-per-worker 1 --memory-limit 4GB


python cleanup3.py --input all-task-318.csv --output dedup_no_clean_3.csv --deduplicate --dedup-field place_id --n-workers 8 --threads-per-worker 1 --memory-limit 4GB

python cleanup3.py --input dedup_no_clean_3.csv --output dedup_clean_3.csv --columns name description reviews competitors owner main_category categories address review_keywords featured_question detailed_address about menu reservations featured_reviews detailed_reviews --clean-line-terminators --remove-control-characters --remove-emojis --normalize-unicode --n-workers 8 --threads-per-worker 1 --memory-limit 4GB


python compare_csv.py clean_dedup_single_step_3.csv clean_dedup_3.csv --key-columns place_id --unique1 unique1.csv --unique2 unique2.csv --differences diffs.csv --log-level DEBUG --log-file compare.log

python compare_csv3.py clean_dedup_single_step_3.csv clean_dedup_3.csv --key-columns place_id --unique1 unique1.csv --unique2 unique2.csv --differences diffs.csv --detailed-differences detailed_diffs.csv --log-level DEBUG --log-file compare.log --n-workers 8 --memory-limit 4GB
```