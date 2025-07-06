from modelscope import snapshot_download

# 指定要下载的 repo 和本地保存位置
snapshot_download(
    repo_id='qwen/Qwen2.5-7B-Instruct',
    local_dir='./models/Qwen2.5-7B-Instruct'
)