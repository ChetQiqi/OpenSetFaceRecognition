python video_benchmark.py \
    --images-dir G:\\YTF_dataset\\OpenDataLab___YouTube_Faces\\raw\\data\\YouTubeFaces\\aligned_images_DB \
    --videos-dir G:\\YTF_dataset\\OpenDataLab___YouTube_Faces\\raw\\data\\YouTubeFaces\\videos \
    --num-persons 100 \
    --random-seed 42 \
    --max-images-per-person 5 \
    --device cuda \
    --db-path benchmark\\YTF_100p.db \
    --output-dir results_100p