import openxlab
openxlab.login(ak="28rmybj7k0kkjkxbgadz", sk="bv8eaqmmgjpzrx5nb1wxwwdwr0ornwo26yzkx7dj") # 进行登录，输入对应的AK/SK，可在个人中心查看AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/YouTube_Faces') #数据集信息查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/YouTube_Faces', target_path='G:\\YTF_dataset') # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/YouTube_Faces',source_path='/README.md', target_path='G:\\YTF_dataset') #数据集文件下载