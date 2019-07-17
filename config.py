import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_list_path = 'jieba_dict/stop_words.txt'
tradition_dict_path = 'jieba_dict/dict.txt.big'
csv_file_path = 'data/data.csv'
SOS_token = 0 #設定初始值的idx
EOS_token = 1 #設定結束符號的idx
MAX_LENGTH = 500 #標題與內文最長設定500字
teacher_forcing_ratio = 0.5
hidden_size = 256