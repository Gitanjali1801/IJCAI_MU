
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

# %%
# %pip install git+https://github.com/openai/CLIP.git

# %%
import torch
import clip
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# %%
image = preprocess(Image.open("/test_images/15001.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# %%
import torch
import clip
# # !pip install git+https://github.com/openai/CLIP.git
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# clip_model, compose = clip.load('ViT-L/14', device = device)
# # text_model = text_model.cpu()
# def process(idx_val,arr):
#   if idx_val=='0':
#     arr.append(0)
#   else:
#     arr.append(1)

# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('/gender_biasness/MAMI_train_scene_graph.csv')
data_test = pd.read_csv('/gender_biasness/MAMI_test_scene_graph.csv')

# %%
from collections import Counter
# %%
data_test

# %%
data.head(10)

# %%
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import clip
from PIL import Image

# %%
# %pip install multilingual-clip torch

# %%
from multilingual_clip import pt_multilingual_clip
import transformers

# %%
texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)

# %%
sample = data['Text_Transcription'][10]

# %%
sample

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# clip_model, compose = clip.load('RN50x4', device = device)
clip_model, compose = clip.load("ViT-B/32", device = device)
text_inputs = (clip.tokenize(data.Text_Transcription.values[321],truncate=True)).to(device)
print(text_inputs)

# %%
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
len(data)

# %%
data.img.values[1:10]

# %%
def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['Text_Transcription'])
  text1 = list(data['rel_label1'])
  text2= list(data['rel_label2'])
  text3 = list(data['rel_label3'])
  text4 = list(data['rel_label4'])
  text5 = list(data['rel_label5'])
  text6 = list(data['rel_label6'])
  text7 = list(data['rel_label7'])
  text8 = list(data['rel_label8'])
  text9 = list(data['rel_label9'])
  text10 = list(data['rel_label10'])
  text11 = list(data['rel_label11'])
  text12 = list(data['rel_label12'])
  text13 = list(data['rel_label13'])
  text14 = list(data['rel_label14'])
  text15 = list(data['rel_label15'])
  text16 = list(data['rel_label16'])
  img_path = list(data['file_name'])
  name = list(data['file_name'])

  # label = list(data['Level1'])
  label = list(data['misogynous'])
  # valence = list(data['Valence'])
  # valence = list(map(lambda x: x - 1 , valence))  
  text_features1=[] 
  text_features2=[]
  text_features3=[]
  text_features4=[]
  text_features5=[]
  text_features6=[]
  text_features7=[]
  text_features8=[]
  text_features9=[]
  text_features10=[]
  text_features11=[]
  text_features12=[]
  text_features13=[]
  text_features14=[]
  text_features15=[]
  text_features16=[]
  # text
  # text2
  # 
  #2
  # rousal = list(data['Arousal'])
  # arousal = list(map(lambda x: x - 1 , arousal))  
  #optimize memory for features
  text_features,image_features,l,Name,v = [],[],[],[],[]
  # for txt,img,L,A,V in tqdm(zip(text,img_path,label,arousal,valence)):
  for txt,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,txt12,txt13,txt14,txt15,txt16,img,L,n in \
    tqdm(zip(text,text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15,text16,img_path,label,name)):
    try:
      #img = preprocess(Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img)).unsqueeze(0).to(device)
      img = Image.open('/training_images/'+img)
    except Exception as e:
      print(e)
      continue

    img = torch.stack([compose(img).to(device)])
    l.append(L)
    Name.append(n)
    # v.append(V)
    #txt = torch.as_tensor(txt)
    with torch.no_grad():
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      temp_txt1=model.forward(txt1, tokenizer).detach().cpu().numpy()
      temp_txt2=model.forward(txt2, tokenizer).detach().cpu().numpy()
      temp_txt3=model.forward(txt3, tokenizer).detach().cpu().numpy()
      temp_txt4=model.forward(txt4, tokenizer).detach().cpu().numpy()
      temp_txt5=model.forward(txt5, tokenizer).detach().cpu().numpy()
      temp_txt6=model.forward(txt6, tokenizer).detach().cpu().numpy()
      temp_txt7=model.forward(txt7, tokenizer).detach().cpu().numpy()
      temp_txt8=model.forward(txt8, tokenizer).detach().cpu().numpy()
      temp_txt9=model.forward(txt9, tokenizer).detach().cpu().numpy()
      temp_txt10=model.forward(txt10, tokenizer).detach().cpu().numpy()
      temp_txt11=model.forward(txt11, tokenizer).detach().cpu().numpy()
      temp_txt12=model.forward(txt12, tokenizer).detach().cpu().numpy()
      temp_txt13=model.forward(txt13, tokenizer).detach().cpu().numpy()
      temp_txt14=model.forward(txt14, tokenizer).detach().cpu().numpy()
      temp_txt15=model.forward(txt15, tokenizer).detach().cpu().numpy()
      temp_txt16=model.forward(txt16, tokenizer).detach().cpu().numpy()
      # temp_tt = model([txt]).detach().cpu().numpy()
      text_features.append(temp_txt)
      text_features1.append(temp_txt1)
      text_features2.append(temp_txt2)
      text_features3.append(temp_txt3)
      text_features4.append(temp_txt4)
      text_features5.append(temp_txt5)
      text_features6.append(temp_txt6)
      text_features7.append(temp_txt7)
      text_features8.append(temp_txt8)
      text_features9.append(temp_txt9)
      text_features10.append(temp_txt10)
      text_features11.append(temp_txt11)
      text_features12.append(temp_txt12)
      text_features13.append(temp_txt13)
      text_features14.append(temp_txt14)
      text_features15.append(temp_txt15)
      text_features16.append(temp_txt16)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img
      
      torch.cuda.empty_cache()
    
    del img
    #del txtla
    torch.cuda.empty_cache()
  return text_features,text_features1,text_features2,text_features3,text_features4,text_features5,text_features6,text_features7,text_features8,text_features9,text_features10,text_features11,text_features12,text_features13\
    ,text_features14,text_features15,text_features16,image_features,l,Name



# %%
# t_f,i_f,label,v,a = get_data(data.head(5))
text_features,text_features1,text_features2,text_features3,text_features4,text_features5,text_features6,text_features7,text_features8,text_features9,text_features10,text_features11,text_features12,text_features13\
    ,text_features14,text_features15,text_features16,image_features,l,Name = get_data(data.head(5))


# %%
outliers = []
for names in tqdm(list(data['file_name'])):
  #change the path according to your drive
  if not os.path.exists('/MAMI_2022_images/training_images/'+names):
    outliers.append(names)

# data = data[~data['Name'].isin(outliers)] 

# %%
outliers

# %%
def get_data_test(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['text'])
  text1 = list(data['rel_label1'])   
  text2= list(data['rel_label2'])
  text3 = list(data['rel_label3'])
  text4 = list(data['rel_label4'])
  text5 = list(data['rel_label5'])
  text6 = list(data['rel_label6'])
  text7 = list(data['rel_label7'])
  text8 = list(data['rel_label8'])
  text9 = list(data['rel_label9'])
  text10 = list(data['rel_label10'])
  text11 = list(data['rel_label11'])
  text12 = list(data['rel_label12'])
  text13 = list(data['rel_label13'])
  text14 = list(data['rel_label14'])
  text15 = list(data['rel_label15'])
  text16 = list(data['rel_label16'])
  img_path = list(data['file_name'])
  name = list(data['file_name'])

  # label = list(data['Level1'])
  label = list(data['label'])
  # valence = list(data['Valence'])
  # valence = list(map(lambda x: x - 1 , valence))  
  text_features1=[] 
  text_features2=[]
  text_features3=[]
  text_features4=[]
  text_features5=[]
  text_features6=[]
  text_features7=[]
  text_features8=[]
  text_features9=[]
  text_features10=[]
  text_features11=[]
  text_features12=[]
  text_features13=[]
  text_features14=[]
  text_features15=[]
  text_features16=[]
  # text
  # text2
  # 
  #2
  # rousal = list(data['Arousal'])
  # arousal = list(map(lambda x: x - 1 , arousal))  
  #optimize memory for features
  text_features,image_features,l,Name,v = [],[],[],[],[]
  # for txt,img,L,A,V in tqdm(zip(text,img_path,label,arousal,valence)):
  for txt,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,txt12,txt13,txt14,txt15,txt16,img,L,n in \
    tqdm(zip(text,text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15,text16,img_path,label,name)):
    try:
      #img = preprocess(Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img)).unsqueeze(0).to(device)
      img = Image.open('/MAMI_2022_images/test_images/'+img)
    except Exception as e:
      print(e)
      continue

    img = torch.stack([compose(img).to(device)])
    l.append(L)
    Name.append(n)
    # v.append(V)
    #txt = torch.as_tensor(txt)
    with torch.no_grad():
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      temp_txt1=model.forward(txt1, tokenizer).detach().cpu().numpy()
      temp_txt2=model.forward(txt2, tokenizer).detach().cpu().numpy()
      temp_txt3=model.forward(txt3, tokenizer).detach().cpu().numpy()
      temp_txt4=model.forward(txt4, tokenizer).detach().cpu().numpy()
      temp_txt5=model.forward(txt5, tokenizer).detach().cpu().numpy()
      temp_txt6=model.forward(txt6, tokenizer).detach().cpu().numpy()
      temp_txt7=model.forward(txt7, tokenizer).detach().cpu().numpy()
      temp_txt8=model.forward(txt8, tokenizer).detach().cpu().numpy()
      temp_txt9=model.forward(txt9, tokenizer).detach().cpu().numpy()
      temp_txt10=model.forward(txt10, tokenizer).detach().cpu().numpy()
      temp_txt11=model.forward(txt11, tokenizer).detach().cpu().numpy()
      temp_txt12=model.forward(txt12, tokenizer).detach().cpu().numpy()
      temp_txt13=model.forward(txt13, tokenizer).detach().cpu().numpy()
      temp_txt14=model.forward(txt14, tokenizer).detach().cpu().numpy()
      temp_txt15=model.forward(txt15, tokenizer).detach().cpu().numpy()
      temp_txt16=model.forward(txt16, tokenizer).detach().cpu().numpy()
      # temp_tt = model([txt]).detach().cpu().numpy()
      text_features.append(temp_txt)
      text_features1.append(temp_txt1)
      text_features2.append(temp_txt2)
      text_features3.append(temp_txt3)
      text_features4.append(temp_txt4)
      text_features5.append(temp_txt5)
      text_features6.append(temp_txt6)
      text_features7.append(temp_txt7)
      text_features8.append(temp_txt8)
      text_features9.append(temp_txt9)
      text_features10.append(temp_txt10)
      text_features11.append(temp_txt11)
      text_features12.append(temp_txt12)
      text_features13.append(temp_txt13)
      text_features14.append(temp_txt14)
      text_features15.append(temp_txt15)
      text_features16.append(temp_txt16)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img
      
      torch.cuda.empty_cache()
    
    del img
    #del txtla
    torch.cuda.empty_cache()
  return text_features,text_features1,text_features2,text_features3,text_features4,text_features5,text_features6,text_features7,text_features8,text_features9,text_features10,text_features11,text_features12,text_features13\
    ,text_features14,text_features15,text_features16,image_features,l,Name



# %%
class HatefulDataset(Dataset):

  def __init__(self,data):
    
    # self.t_f,self.i_f,self.label,self.v,self.a = get_data(data)
    self.t_f,self.t_f1,self.t_f2,self.t_f3,self.t_f4,self.t_f5,self.t_f6,self.t_f7,self.t_f8,self.t_f9,self.t_f10,self.t_f11,self.t_f12,self.t_f13,self.t_f14,self.t_f15,self.t_f16,self.i_f,self.label,self.name =get_data(data)
    # self.t_f,self.i_f,self.label,self.name = get_data(data)
    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.t_f1 = np.squeeze(np.asarray(self.t_f1),axis=1)
    self.t_f2 = np.squeeze(np.asarray(self.t_f2),axis=1)
    self.t_f3 = np.squeeze(np.asarray(self.t_f3),axis=1)
    self.t_f4 = np.squeeze(np.asarray(self.t_f4),axis=1)
    self.t_f5 = np.squeeze(np.asarray(self.t_f5),axis=1)
    self.t_f6 = np.squeeze(np.asarray(self.t_f6),axis=1)
    self.t_f7 = np.squeeze(np.asarray(self.t_f7),axis=1)
    self.t_f8 = np.squeeze(np.asarray(self.t_f8),axis=1)
    self.t_f9 = np.squeeze(np.asarray(self.t_f9),axis=1)
    self.t_f10 = np.squeeze(np.asarray(self.t_f10),axis=1)
    self.t_f11 = np.squeeze(np.asarray(self.t_f11),axis=1)
    self.t_f12 = np.squeeze(np.asarray(self.t_f12),axis=1)
    self.t_f13 = np.squeeze(np.asarray(self.t_f13),axis=1)
    self.t_f14 = np.squeeze(np.asarray(self.t_f14),axis=1)
    self.t_f15 = np.squeeze(np.asarray(self.t_f15),axis=1)
    self.t_f16 = np.squeeze(np.asarray(self.t_f16),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)

    
    
  def __len__(self):
    return len(self.label)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    name=self.name[idx]
    label = self.label[idx]
    T = self.t_f[idx,:]
    T1 = self.t_f1[idx,:]
    T2 = self.t_f2[idx,:]
    T3 = self.t_f3[idx,:]
    T4 = self.t_f4[idx,:]
    T5 = self.t_f5[idx,:]
    T6 = self.t_f6[idx,:]
    T7 = self.t_f7[idx,:]
    T8 = self.t_f8[idx,:]
    T9 = self.t_f9[idx,:]
    T10 = self.t_f10[idx,:]
    T11 = self.t_f11[idx,:]
    T12= self.t_f12[idx,:]
    T13 = self.t_f13[idx,:]
    T14 = self.t_f14[idx,:]
    T15 = self.t_f15[idx,:]
    T16 = self.t_f16[idx,:]
    I = self.i_f[idx,:]
    
    # v = self.v[idx]
    # a = self.a[idx]
    # sample = {'label':label,'processed_txt':T,'processed_img':I,'valence':v,'arousal':a}
    sample = {'label':label,'processed_txt':T,'processed_txt1':T1,'processed_txt2':T2,'processed_txt3':T3,'processed_txt4':T4,'processed_txt5':T5,'processed_txt6':T6,'processed_txt7':T7,'processed_txt8':T8,\
              'processed_txt9':T9,'processed_txt10':T10,'processed_txt11':T11,'processed_txt12':T12,'processed_txt13':T13,'processed_txt14':T14,'processed_txt15':T15,'processed_txt16':T16,'processed_img':I,'name':name}
    return sample
    

# %%
len(data)

# %%
sample_dataset = HatefulDataset(data)

# %%
torch.save(sample_dataset,'MAMI_rel_label_train.pt')


# %%
test= torch.load("MAMI_rel_label_test.pt")

# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        #self.__C = __C
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)
        
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        
        exp_out = img_feat * ques_feat             # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z
# %%
data = data[~data['Name'].isin(outliers)]

# %%
# %cd /gender_biasness

# %%
sample_dataset_new= torch.load("/gender_biasness/MAMI_rel_label.pt")

len(sample_dataset_new)

# %%
torch.manual_seed(123)
# t_p,v_p = torch.utils.data.random_split(sample_dataset_new,[9000,941])
t_p,v_p = torch.utils.data.random_split(sample_dataset_new,[9000,941])

# torch.manual_seed(123)
# t_p,te_p = torch.utils.data.random_split(t_p,[500,400])

# %%
t_p[1]["processed_img"].shape
import torch
import torch.nn as nn

class BiLSTMAttention_img(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMAttention_img, self).__init__()
        self.bilstm = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.linear_layer = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim*2, 1)
        # self.output_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        self.output_layer = torch.nn.Linear(832, 64)


    def forward(self, x, additional_tensor):
        # Pass input tensor through BiLSTM
        lstm_output, _ = self.bilstm(x)

        # Calculate attention weights
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=0)

        # Apply attention to BiLSTM output
        attentive_representation = torch.sum(attention_weights * lstm_output, dim=0)

        # Apply linear layer to BiLSTM output
        lstm_output = self.linear_layer(lstm_output)

        # Squeeze dimensions
        lstm_output = lstm_output.squeeze(1)
        attentive_representation = attentive_representation.squeeze(1)

        # Concatenate attentive representation with additional tensor
        concatenated_output = torch.cat((lstm_output, attentive_representation.expand(lstm_output.size(0), -1)), dim=1)
        # print("concatenated_output",concatenated_output.shape)
                # Add the additional tensor
        # print(concatenated_output)
        # print("addtional tensor",additional_tensor)
        # print('additional_tensor',additional_tensor.shape)
        # concatenated_output += additional_tensor
        concatenated = torch.cat([concatenated_output,additional_tensor], dim=1)

        # print(concatenated)
        # print("concatenated",concatenated.shape)
        # test=concatenated_output.unsqueeze(1)
        # print("test", test.shape)
        # Pass concatenated output through the output layer
        output = self.output_layer(concatenated)
        # output = self.output_layer(concatenated_output.unsqueeze(1))
        # print("output.shape",output.shape)
        return output

# %%
import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMAttention, self).__init__()
        self.bilstm = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.linear_layer = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim*2, 1)
        # self.output_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        self.output_layer = torch.nn.Linear(832, 64)


    def forward(self, x, additional_tensor):
        # Pass input tensor through BiLSTM
        lstm_output, _ = self.bilstm(x)

        # Calculate attention weights
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=0)

        # Apply attention to BiLSTM output
        attentive_representation = torch.sum(attention_weights * lstm_output, dim=0)

        # Apply linear layer to BiLSTM output
        lstm_output = self.linear_layer(lstm_output)

        # Squeeze dimensions
        lstm_output = lstm_output.squeeze(1)
        attentive_representation = attentive_representation.squeeze(1)

        # Concatenate attentive representation with additional tensor
        concatenated_output = torch.cat((lstm_output, attentive_representation.expand(lstm_output.size(0), -1)), dim=1)
        # print("concatenated_output",concatenated_output.shape)
                # Add the additional tensor
        # print(concatenated_output)
        # print("addtional tensor",additional_tensor)
        # print('additional_tensor',additional_tensor.shape)
        # concatenated_output += additional_tensor
        concatenated = torch.cat([concatenated_output,additional_tensor], dim=1)

        # print(concatenated)
        # print("concatenated",concatenated.shape)
        # test=concatenated_output.unsqueeze(1)
        # print("test", test.shape)
        # Pass concatenated output through the output layer
        output = self.output_layer(concatenated)
        # output = self.output_layer(concatenated_output.unsqueeze(1))
        # print("output.shape",output.shape)
        return output


# %%
# %cd /gender_biasness

# %%
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
class Classifier(pl.LightningModule):

  def __init__(self):
    super().__init__()
    # self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    # self.MFB = MFB(640,640,True,256,64,0.1)
    self.MFB = MFB(512,768,True,256,64,0.1)
    # self.memory_network = MemoryNetwork(input_dim=512, hidden_dim=256, output_dim=2)
    self.model = BiLSTMAttention(input_dim=768, hidden_dim=256, output_dim=64)   #BiLSTMAttention_img
    # self.model_img = BiLSTMAttention_img(input_dim=512, hidden_dim=256, output_dim=64)   #BiLSTMAttention_img

    # self.memory_network = MemoryNetwork(input_dim=768, hidden_dim=128, output_dim=2)
    self.fin_old = torch.nn.Linear(64*3,2)
    self.fin_old_object = torch.nn.Linear(64,2)
    self.fin = torch.nn.Linear(16 * 768, 64)
    # self.fin = torch.nn.Linear(2048,2)
    # self.fin = torch.nn.Linear(768, 2)
    # self.fin = torch.nn.Linear(80, 2)  # Adjust the input dimension as per concatenation
    # self.pooling = nn.AvgPool1d(kernel_size=concatenated_vector.shape[1])  # Apply average pooling

  def forward(self, x,y,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16):
      x_,y_ = x,y
      #print(x_.size())
      #x = torch.cat((x,nrc),1)
      #print(c.size())
      # print(x.shape)
      # # print(y.shape)
      x = x.float()
      y = y.float()
      # x = x.float().to(device)
      # y = y.float().to(device)
      z = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      # c = self.fin(torch.squeeze(z,dim=1))
      vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]
      # Compute cosine similarities
      # vectors = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16]
      cosine_similarities = [torch.cosine_similarity(vector, x, dim=0) for vector in vectors]

      # Calculate weights using softmax
      weights = F.softmax(torch.cat(cosine_similarities, dim=0), dim=0)
      # Apply weights to vectors and concatenate
      weighted_vectors = [weight * vector for weight, vector in zip(weights, vectors)]
      # concatenated_vector = torch.cat(weighted_vectors, dim=0)
      # print("concatenated_vector.shape",concatenated_vector.shape)
      concatenated_vector = torch.cat(weighted_vectors, dim=1)
      # Flatten the concatenated vector
      flattened_vector = concatenated_vector.view(concatenated_vector.size(0), -1)
      # print("flattened_vector.shape",flattened_vector.shape)
      # Pass flattened concatenated vector through linear layer
      output = self.fin(flattened_vector)
      # print("output.shape",output.shape)
      # print("z.shape", z.shape)
      z_new=torch.squeeze(z,dim=1)
      # print("reshape z",z_new.shape)
      print("z_new.shape",z_new.shape)
      print("x.shape",x.shape)
      memory = self.model(torch.unsqueeze(x,axis=1), z_new)
      print("memory.shape",memory.shape)
      # memory_img = self.model_img(torch.unsqueeze(y,axis=1), z_new)
      # print("reshape z",z_new.shape)
      # c = self.fin_old(torch.squeeze(z,dim=1))
      concat_new = torch.cat([output,z_new,memory], dim=1)
      print("concat_new.shape",concat_new.shape)

      # print("concat_new.shape",concat_new.shape)
      # flattened = concat_new.view(concat_new.size(0), -1)
      # print("flattened.shape",flattened.shape)
      c = self.fin_old(concat_new)
      print("c.shape",c.shape)
      # c_object = self.fin_old_object(output)
      # Probability distribution over labels
      # output = torch.log_softmax(output, dim=1)
      output1 = torch.log_softmax(c, dim=1)
      # output_object = torch.log_softmax(c_object, dim=1)
      return output1,memory,output


  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  
  def KLloss(self, ten1, ten2):
     kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
     return kl_loss(ten1,ten2)

  def training_step(self, train_batch, batch_idx):
      # lab,txt,img,v,a,_,_,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12 = train_batch
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= train_batch

      lab = train_batch[lab]
      #print(lab)
      txt = train_batch[txt]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      e6 = train_batch[e6]
      e7 = train_batch[e7]
      e8 = train_batch[e8]
      e9 = train_batch[e9]
      e10 = train_batch[e10]
      e11 = train_batch[e11]
      e12 = train_batch[e12]
      e13 = train_batch[e13]
      e14 = train_batch[e14]
      e15 = train_batch[e15]
      e16 = train_batch[e16]
      #4rint(txt4
      img = train_batch[img]
      logit_offen,z_new,output= self.forward(txt,img,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16)
      # print(logit_offen)
      # loss = self.cross_entropy_loss(logit_offen, lab)
      # q_sentiment = self.memory_network.query(txt)
      # print("q_sentiment",q_sentiment.shape)
      # output = self.memory_network(txt, q_sentiment)
      loss2 = self.KLloss(z_new, output)
      # print(loss2)
      # logit_offen = self.forward(txt, img)
      loss1 = self.cross_entropy_loss(logit_offen, lab)
      # loss1 = self.cross_entropy_loss(logit_offen, lab)

      # loss2 = self.cross_entropy_loss(output, lab)
      loss = (0.7*loss1)+(0.3*loss2)
      # loss = (0.6*loss1)+(0.4*loss2)
      #loss = loss1+loss3
      self.log('train_loss', loss)
      return loss


  def validation_step(self, val_batch, batch_idx):
      # lab,txt,img,name= val_batch
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= val_batch

      lab = val_batch[lab]
      #print(lab)
      txt = val_batch[txt]
      e1 = val_batch[e1]
      e2 = val_batch[e2]
      e3 = val_batch[e3]
      e4 = val_batch[e4]
      e5 = val_batch[e5]
      e6 = val_batch[e6]
      e7 = val_batch[e7]
      e8 = val_batch[e8]
      e9 = val_batch[e9]
      e10 = val_batch[e10]
      e11 = val_batch[e11]
      e12 = val_batch[e12]
      e13 = val_batch[e13]
      e14 = val_batch[e14]
      e15 = val_batch[e15]
      e16 = val_batch[e16]
      #4rint(txt4
      img = val_batch[img]
      logits,z_new,output= self.forward(txt,img,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16)
      # lab = val_batch[lab]
      # txt = val_batch[txt]
      # img = val_batch[img]
      # logits = self.forward(txt,img)
      loss2 = self.KLloss(z_new, output)

      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss1 = self.cross_entropy_loss(logits, lab)
      # loss=loss1+loss2

      loss = (0.7*loss1)+(0.3*loss2)
      # loss = (0.6*loss1)+(0.4*loss2)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      return {
                'progress_bar': tqdm_dict,
      'val_f1 offensive': f1_score(lab,tmp,average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs14.append(out['val_f1 offensive'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_f1 offensive', sum(outs14)/len(outs14))
    print(f'***val_acc_all_offn at epoch end {sum(outs)/len(outs)}****')
    print(f'***val_f1 offensive at epoch end {sum(outs14)/len(outs14)}****')
  
  def test_step(self, batch, batch_idx):
      # lab,txt,img,name= batch
            # lab,txt,img,name= val_batch
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= batch

      lab = batch[lab]
      #print(lab)
      txt = batch[txt]
      e1 = batch[e1]
      e2 = batch[e2]
      e3 = batch[e3]
      e4 = batch[e4]
      e5 = batch[e5]
      e6 = batch[e6]
      e7 = batch[e7]
      e8 = batch[e8]
      e9 = batch[e9]
      e10 = batch[e10]
      e11 = batch[e11]
      e12 = batch[e12]
      e13 = batch[e13]
      e14 = batch[e14]
      e15 = batch[e15]
      e16 = batch[e16]
      #4rint(txt4
      img = batch[img]
      logits,z_new,output= self.forward(txt,img,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16)
      # lab = batch[lab]
      # txt = batch[txt]
      # img = batch[img]
      # logits = self.forward(txt,img)
      loss2 = self.KLloss(z_new, output)
      # logit_new= (logits+logit_offen_object)/2

      # tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)

      # print(tmp_base)
  
      # tmp_obj = np.argmax(logit_offen_object.detach().cpu().numpy(),axis=-1)
      # print(tmp_obj)

      # tmp= (tmp_base+tmp_obj)/2
      # loss1 = self.cross_entropy_loss(logits, lab)
      loss1 = self.cross_entropy_loss(logits, lab)

      # loss=loss1+loss2
      loss = (0.7*loss1)+(0.3*loss2)
      # loss = (0.6*loss1)+(0.4*loss2)
    

      lab = lab.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test_roc_auc',roc_auc_score(lab,tmp))
      # self.log('confusion matrix',confusion_matrix(lab,tmp))

# print(cm)
      self.log('test_loss', loss)
      tqdm_dict = {'test_acc': accuracy_score(lab,tmp)}
      #print('Val acc {}'.format(accuracy_score(lab,tmp)))
      return {
                'progress_bar': tqdm_dict,
                'test_acc': accuracy_score(lab,tmp),
                'test_f1_score': f1_score(lab,tmp,average='macro'),
      }
  def test_epoch_end(self, outputs):
      # OPTIONAL
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[]
      for out in outputs:
        # outs15.append(out['test_loss_target'])
        outs.append(out['test_acc'])
        outs2.append(out['test_f1_score'])
      self.log('test_acc', sum(outs)/len(outs))
      self.log('test_f1_score', sum(outs2)/len(outs2))

  def configure_optimizers(self):
    # optimizer = torch.optim.Adam(self.parameters(), lr=3e-2)
    # optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    return optimizer


class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):  
    self.hm_train = t_p
    self.hm_val = v_p
    # self.hm_test = te_p
    self.hm_test = test   #

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=128)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=128)
  
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()
checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='ckpts_mami/',
     filename='epoch{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
# train
from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
hm_model = Classifier()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,deterministic=True,max_epochs=2,precision=16,callbacks=all_callbacks)
# trainer = pl.Trainer(gpus=gpus,max_epochs=10,callbacks=all_callbacks)
trainer.fit(hm_model, data_module)


# %%
print(np.__version__)


# %%
model_pre_trained = hm_model.load_from_checkpoint('/ckpts_mami/epoch48-val_f1_all_offn.ckpt')
model_pre_trained.to(device)
# model.freeze()

# %%
# test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
test_dataloader = DataLoader(dataset=test, batch_size=1478)
ckpt_path = '/ckpts_mami/epoch48-val_f1_all_offn.ckpt' # put ckpt_path according to the path output in the previous cell
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)

# %%
#from lime.lime_text import LimeTextExplainer
from lime import lime_image
label_names = [i for i in range(2)]
#explainer = lime.lime_text.LimeTextExplainer(class_names=label_names)
explainer = lime_image.LimeImageExplainer()

# %%
name = list(data_test['file_name'])

# %%
name

# %%
name[90]

# %%
def f1(img):
  #img = txt_to_img[txt]
  #img = Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img_name)

  #img = Image.open('/content/train_images/'+img)
  #img = torch.stack([compose(img).to(device)])
  print(img.shape)
  txt = list(data_test.loc[data_test['file_name'] == name[90]]['text'])[0]
  text1 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label1'])[0]   
  text2= list(data_test.loc[data_test['file_name'] == name[90]]['rel_label2'])[0]
  text3 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label3'])[0]
  text4 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label4'])[0]
  text5 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label5'])[0]
  text6 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label6'])[0]
  text7 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label7'])[0]
  text8 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label8'])[0]
  text9 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label9'])[0]
  text10 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label10'])[0]
  text11 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label11'])[0]
  text12 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label12'])[0]
  text13 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label13'])[0]
  text14 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label14'])[0]
  text15 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label15'])[0]
  text16 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label16'])[0]
  
  k = []
  for i in range(img.shape[0]):
    img_t = Image.fromarray(img[i,:,:,:].astype('uint8'), 'RGB')
    img_t = torch.stack([compose(img_t).to(device)])
    # k.append(torch.squeeze(img_t).numpy())
    k.append(torch.squeeze(img_t).cpu().numpy())
    #print(img_t.shape)
  fin = 0
  #print(k)
  img = torch.tensor(np.asarray(k))
  print(txt)
  with torch.no_grad():
    # with torch.no_grad():
    temp_img = clip_model.encode_image(img.to(device)).detach().cpu().numpy()
    # temp_img = clip_model.encode_image(img).detach().cpu().numpy()
    temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
    temp_txt1=model.forward(text1, tokenizer).detach().cpu().numpy()
    temp_txt2=model.forward(text2, tokenizer).detach().cpu().numpy()
    temp_txt3=model.forward(text3, tokenizer).detach().cpu().numpy()
    temp_txt4=model.forward(text4, tokenizer).detach().cpu().numpy()
    temp_txt5=model.forward(text5, tokenizer).detach().cpu().numpy()
    temp_txt6=model.forward(text6, tokenizer).detach().cpu().numpy()
    temp_txt7=model.forward(text7, tokenizer).detach().cpu().numpy()
    temp_txt8=model.forward(text8, tokenizer).detach().cpu().numpy()
    temp_txt9=model.forward(text9, tokenizer).detach().cpu().numpy()
    temp_txt10=model.forward(text10, tokenizer).detach().cpu().numpy()
    temp_txt11=model.forward(text11, tokenizer).detach().cpu().numpy()
    temp_txt12=model.forward(text12, tokenizer).detach().cpu().numpy()
    temp_txt13=model.forward(text13, tokenizer).detach().cpu().numpy()
    temp_txt14=model.forward(text14, tokenizer).detach().cpu().numpy()
    temp_txt15=model.forward(text15, tokenizer).detach().cpu().numpy()
    temp_txt16=model.forward(text16, tokenizer).detach().cpu().numpy()
    txt,img = torch.tensor(temp_txt).to(device),torch.tensor(temp_img).to(device)
    txt1=torch.tensor(temp_txt1).to(device)
    txt2=torch.tensor(temp_txt2).to(device)
    txt3=torch.tensor(temp_txt3).to(device)
    txt4=torch.tensor(temp_txt4).to(device)
    txt5=torch.tensor(temp_txt5).to(device)
    txt6=torch.tensor(temp_txt6).to(device)
    txt7=torch.tensor(temp_txt7).to(device)
    txt8=torch.tensor(temp_txt8).to(device)
    txt9=torch.tensor(temp_txt9).to(device)
    txt10=torch.tensor(temp_txt10).to(device)
    txt11=torch.tensor(temp_txt11).to(device)
    txt12=torch.tensor(temp_txt12).to(device)
    txt13=torch.tensor(temp_txt13).to(device)
    txt14=torch.tensor(temp_txt14).to(device)
    txt15=torch.tensor(temp_txt15).to(device)
    txt16=torch.tensor(temp_txt16).to(device)
    #img = torch.cat(500*[img])
    txt = torch.cat(10*[txt])
    txt1=torch.cat(10*[txt1])
    txt2=torch.cat(10*[txt2])
    txt3=torch.cat(10*[txt3])
    txt4=torch.cat(10*[txt4])
    txt5=torch.cat(10*[txt5])
    txt6=torch.cat(10*[txt6])
    txt7=torch.cat(10*[txt7])
    txt8=torch.cat(10*[txt8])
    txt9=torch.cat(10*[txt9])
    txt10=torch.cat(10*[txt10])
    txt11=torch.cat(10*[txt11])
    txt12=torch.cat(10*[txt12])
    txt13=torch.cat(10*[txt13])
    txt14=torch.cat(10*[txt14])
    txt15=torch.cat(10*[txt15])
    txt16=torch.cat(10*[txt16])
    print(txt.size(),img.size())
    # hidden_pred,z_new,output = model_pre_trained(txt,img,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,txt12,txt13,txt14,txt15,txt16)
    hidden_pred = model_pre_trained(txt,img)

    print(hidden_pred.size())
    fin = torch.nn.Softmax(dim=1)(hidden_pred)

    del txt
    del img
  return fin.cpu().numpy()
def get_explanation(idx):
  #idx = 9
  txt = list(data_test.loc[data_test['file_name'] == name[idx]]['text'])[0]
  img_name = name[idx]
  img = Image.open('/gender_biasness/MAMI_2022_images/test_images/'+img_name)
  explanation = explainer.explain_instance(np.array(img), 
                                         f1, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=100)
  from skimage.segmentation import mark_boundaries
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
  img_boundry2 = mark_boundaries(temp/255.0, mask)
  plt.imshow(img_boundry2)
get_explanation(90)



# %%
from lime.lime_tabular import LimeTabularExplainer

# %%
def f2(txt):
  #img = txt_to_img[txt]
  img_name = name[9]
  img = Image.open('/gender_biasness/MAMI_2022_images/test_images/'+img_name)
  img = torch.stack([compose(img).to(device)])
  print(img.shape)
  txt = list(data_test.loc[data_test['file_name'] == name[90]]['text'])[0]
  text1 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label1'])[0]   
  text2= list(data_test.loc[data_test['file_name'] == name[90]]['rel_label2'])[0]
  text3 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label3'])[0]
  text4 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label4'])[0]
  text5 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label5'])[0]
  text6 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label6'])[0]
  text7 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label7'])[0]
  text8 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label8'])[0]
  text9 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label9'])[0]
  text10 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label10'])[0]
  text11 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label11'])[0]
  text12 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label12'])[0]
  text13 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label13'])[0]
  text14 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label14'])[0]
  text15 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label15'])[0]
  text16 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label16'])[0]
  fin = 0
  print(txt)
  txt="I hate this bitch. She is in kitchen"
  with torch.no_grad():
    # with torch.no_grad():
    temp_img = clip_model.encode_image(img.to(device)).detach().cpu().numpy()
    # temp_img = clip_model.encode_image(img).detach().cpu().numpy()
    temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
    temp_txt1=model.forward(text1, tokenizer).detach().cpu().numpy()
    temp_txt2=model.forward(text2, tokenizer).detach().cpu().numpy()
    temp_txt3=model.forward(text3, tokenizer).detach().cpu().numpy()
    temp_txt4=model.forward(text4, tokenizer).detach().cpu().numpy()
    temp_txt5=model.forward(text5, tokenizer).detach().cpu().numpy()
    temp_txt6=model.forward(text6, tokenizer).detach().cpu().numpy()
    temp_txt7=model.forward(text7, tokenizer).detach().cpu().numpy()
    temp_txt8=model.forward(text8, tokenizer).detach().cpu().numpy()
    temp_txt9=model.forward(text9, tokenizer).detach().cpu().numpy()
    temp_txt10=model.forward(text10, tokenizer).detach().cpu().numpy()
    temp_txt11=model.forward(text11, tokenizer).detach().cpu().numpy()
    temp_txt12=model.forward(text12, tokenizer).detach().cpu().numpy()
    temp_txt13=model.forward(text13, tokenizer).detach().cpu().numpy()
    temp_txt14=model.forward(text14, tokenizer).detach().cpu().numpy()
    temp_txt15=model.forward(text15, tokenizer).detach().cpu().numpy()
    temp_txt16=model.forward(text16, tokenizer).detach().cpu().numpy()
    txt,img = torch.tensor(temp_txt).to(device),torch.tensor(temp_img).to(device)
    txt1=torch.tensor(temp_txt1).to(device)
    txt2=torch.tensor(temp_txt2).to(device)
    txt3=torch.tensor(temp_txt3).to(device)
    txt4=torch.tensor(temp_txt4).to(device)
    txt5=torch.tensor(temp_txt5).to(device)
    txt6=torch.tensor(temp_txt6).to(device)
    txt7=torch.tensor(temp_txt7).to(device)
    txt8=torch.tensor(temp_txt8).to(device)
    txt9=torch.tensor(temp_txt9).to(device)
    txt10=torch.tensor(temp_txt10).to(device)
    txt11=torch.tensor(temp_txt11).to(device)
    txt12=torch.tensor(temp_txt12).to(device)
    txt13=torch.tensor(temp_txt13).to(device)
    txt14=torch.tensor(temp_txt14).to(device)
    txt15=torch.tensor(temp_txt15).to(device)
    txt16=torch.tensor(temp_txt16).to(device)
    print(txt.size(),img.size())
    img = torch.cat(200*[img])
    # txt = torch.cat(200*[txt])

    # logits,z_new,output= self.forward(txt,img,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16)

    # hidden_pred,z_new,output = model_pre_trained(txt,img,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,txt12,txt13,txt14,txt15,txt16)
    hidden_pred= model_pre_trained(txt,img)

    print(hidden_pred.size())
    # _,logit_offen,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = hm_model(txt,img,hidden_pred)
  #logit_inten_fin = inten.detach().cpu().numpy()
    fin = torch.nn.Softmax(dim=1)(hidden_pred)
    # probs5 = fin.topk(5)
    # preds = np.array([pred / pred.sum() for pred in fin])
    # print(fin)
    del txt
    del img
  return fin.cpu().numpy()

#txt = list(data.loc[data['Name'] == name[1]]['text'])[0]
def get_explanation_txt(idx):
  #idx = 9
    #idx = 9
  txt = list(data_test.loc[data_test['file_name'] == name[idx]]['text'])[0]
  print(txt)
  txt="I hate this bitch. She is in kitchen"
  print(txt)
  img_name = name[idx]
  # txt = list(data.loc[data['Name'] == name[idx]]['text'])[0]
  # img_name = name[idx]
  # print(img)
  #img = torch.stack([compose(img).to(device)])
  #print(img.shape)
  #s = sar[idx].numpy()
  #print(s)
  #exp = explainer.explain_instance(img, f1, num_features=6, num_samples=500)
  from lime.lime_text import LimeTextExplainer
  # txt_explainer = LimeTabularExplainer(pd.DataFrame([txt]), feature_names=['text'], class_names=[i for i in range(2)])
  txt_explainer = LimeTextExplainer(class_names=[i for i in range(2)])
  explanation = txt_explainer.explain_instance(txt, 
                                         f2, # classification function
                                         top_labels=2, 
                                         num_features=10,
                                         num_samples=200)
  words = explanation.as_list()
  print(words)
  explanation.show_in_notebook(text=True,predict_proba=True, show_predicted_value=True)
  # explanation.show_in_notebook(txt)

#get text explanation  
get_explanation_txt(90)

# %%
# Crete classifier to check shap model
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
class Classifier(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.MFB = MFB(512,768,True,256,64,0.1)
    self.fin = torch.nn.Linear(64, 2)
  def forward(self, x,y):
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      z = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      c = self.fin(torch.squeeze(z,dim=1))
      output1 = torch.log_softmax(c, dim=1)
      return output1


  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  
  def KLloss(self, ten1, ten2):
     kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
     return kl_loss(ten1,ten2)

  def training_step(self, train_batch, batch_idx):
      # lab,txt,img,v,a,_,_,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12 = train_batch
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= train_batch

      lab = train_batch[lab]
      #print(lab)
      txt = train_batch[txt]
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      e6 = train_batch[e6]
      e7 = train_batch[e7]
      e8 = train_batch[e8]
      e9 = train_batch[e9]
      e10 = train_batch[e10]
      e11 = train_batch[e11]
      e12 = train_batch[e12]
      e13 = train_batch[e13]
      e14 = train_batch[e14]
      e15 = train_batch[e15]
      e16 = train_batch[e16]
      #4rint(txt4
      img = train_batch[img]
      logit_offen= self.forward(txt,img)
      loss = self.cross_entropy_loss(logit_offen, lab)
      self.log('train_loss', loss)
      return loss


  def validation_step(self, val_batch, batch_idx):
      # lab,txt,img,name= val_batch
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= val_batch

      lab = val_batch[lab]
      #print(lab)
      txt = val_batch[txt]
      e1 = val_batch[e1]
      e2 = val_batch[e2]
      e3 = val_batch[e3]
      e4 = val_batch[e4]
      e5 = val_batch[e5]
      e6 = val_batch[e6]
      e7 = val_batch[e7]
      e8 = val_batch[e8]
      e9 = val_batch[e9]
      e10 = val_batch[e10]
      e11 = val_batch[e11]
      e12 = val_batch[e12]
      e13 = val_batch[e13]
      e14 = val_batch[e14]
      e15 = val_batch[e15]
      e16 = val_batch[e16]
      #4rint(txt4
      img = val_batch[img]
      logits= self.forward(txt,img)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      return {
                'progress_bar': tqdm_dict,
      'val_f1 offensive': f1_score(lab,tmp,average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs14.append(out['val_f1 offensive'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_f1 offensive', sum(outs14)/len(outs14))
    print(f'***val_acc_all_offn at epoch end {sum(outs)/len(outs)}****')
    print(f'***val_f1 offensive at epoch end {sum(outs14)/len(outs14)}****')
  
  def test_step(self, batch, batch_idx):
      lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= batch

      lab = batch[lab]
      #print(lab)
      txt = batch[txt]
      e1 = batch[e1]
      e2 = batch[e2]
      e3 = batch[e3]
      e4 = batch[e4]
      e5 = batch[e5]
      e6 = batch[e6]
      e7 = batch[e7]
      e8 = batch[e8]
      e9 = batch[e9]
      e10 = batch[e10]
      e11 = batch[e11]
      e12 = batch[e12]
      e13 = batch[e13]
      e14 = batch[e14]
      e15 = batch[e15]
      e16 = batch[e16]
      #4rint(txt4
      img = batch[img]
      logits= self.forward(txt,img)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test_roc_auc',roc_auc_score(lab,tmp))
      self.log('test_loss', loss)
      tqdm_dict = {'test_acc': accuracy_score(lab,tmp)}
      return {
                'progress_bar': tqdm_dict,
                'test_acc': accuracy_score(lab,tmp),
                'test_f1_score': f1_score(lab,tmp,average='macro'),
      }
  def test_epoch_end(self, outputs):
      # OPTIONAL
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[]
      for out in outputs:
        # outs15.append(out['test_loss_target'])
        outs.append(out['test_acc'])
        outs2.append(out['test_f1_score'])
      self.log('test_acc', sum(outs)/len(outs))
      self.log('test_f1_score', sum(outs2)/len(outs2))

  def configure_optimizers(self):
    # optimizer = torch.optim.Adam(self.parameters(), lr=3e-2)
    # optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)

    return optimizer


class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):  
    self.hm_train = t_p
    self.hm_val = v_p
    # self.hm_test = te_p
    self.hm_test = test   #

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=128)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=128)
  
  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()
checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='ckpts_shap/',
     filename='epoch{epoch:02d}-val_f1_all_offn{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
# train
from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
hm_model = Classifier()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,deterministic=True,max_epochs=2,precision=16,callbacks=all_callbacks)
# trainer.fit(hm_model, data_module)


# %%
model_pre_trained = hm_model.load_from_checkpoint('/home/epoch.ckpt')
model_pre_trained.to(device)
# model.freeze()

# %%
import shap

# %%
#img = txt_to_img[txt]
name = list(data_test['file_name'])
img_name = name[9]
img = Image.open('/MAMI_2022_images/test_images/'+img_name)
img = torch.stack([compose(img).to(device)])
print(img.shape)
txt = list(data_test.loc[data_test['file_name'] == name[90]]['text'])[0]
text1 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label1'])[0]   
text2= list(data_test.loc[data_test['file_name'] == name[90]]['rel_label2'])[0]
text3 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label3'])[0]
text4 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label4'])[0]
text5 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label5'])[0]
text6 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label6'])[0]
text7 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label7'])[0]
text8 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label8'])[0]
text9 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label9'])[0]
text10 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label10'])[0]
text11 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label11'])[0]
text12 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label12'])[0]
text13 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label13'])[0]
text14 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label14'])[0]
text15 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label15'])[0]
text16 = list(data_test.loc[data_test['file_name'] == name[90]]['rel_label16'])[0]
fin = 0
print(txt)
with torch.no_grad():
  # with torch.no_grad():
  temp_img = clip_model.encode_image(img.to(device)).detach().cpu().numpy()
  # temp_img = clip_model.encode_image(img).detach().cpu().numpy()
  temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
  temp_txt1=model.forward(text1, tokenizer).detach().cpu().numpy()
  temp_txt2=model.forward(text2, tokenizer).detach().cpu().numpy()
  temp_txt3=model.forward(text3, tokenizer).detach().cpu().numpy()
  temp_txt4=model.forward(text4, tokenizer).detach().cpu().numpy()
  temp_txt5=model.forward(text5, tokenizer).detach().cpu().numpy()
  temp_txt6=model.forward(text6, tokenizer).detach().cpu().numpy()
  temp_txt7=model.forward(text7, tokenizer).detach().cpu().numpy()
  temp_txt8=model.forward(text8, tokenizer).detach().cpu().numpy()
  temp_txt9=model.forward(text9, tokenizer).detach().cpu().numpy()
  temp_txt10=model.forward(text10, tokenizer).detach().cpu().numpy()
  temp_txt11=model.forward(text11, tokenizer).detach().cpu().numpy()
  temp_txt12=model.forward(text12, tokenizer).detach().cpu().numpy()
  temp_txt13=model.forward(text13, tokenizer).detach().cpu().numpy()
  temp_txt14=model.forward(text14, tokenizer).detach().cpu().numpy()
  temp_txt15=model.forward(text15, tokenizer).detach().cpu().numpy()
  temp_txt16=model.forward(text16, tokenizer).detach().cpu().numpy()
  txt,img = torch.tensor(temp_txt).to(device),torch.tensor(temp_img).to(device)
  txt1=torch.tensor(temp_txt1).to(device)
  txt2=torch.tensor(temp_txt2).to(device)
  txt3=torch.tensor(temp_txt3).to(device)
  txt4=torch.tensor(temp_txt4).to(device)
  txt5=torch.tensor(temp_txt5).to(device)
  txt6=torch.tensor(temp_txt6).to(device)
  txt7=torch.tensor(temp_txt7).to(device)
  txt8=torch.tensor(temp_txt8).to(device)
  txt9=torch.tensor(temp_txt9).to(device)
  txt10=torch.tensor(temp_txt10).to(device)
  txt11=torch.tensor(temp_txt11).to(device)
  txt12=torch.tensor(temp_txt12).to(device)
  txt13=torch.tensor(temp_txt13).to(device)
  txt14=torch.tensor(temp_txt14).to(device)
  txt15=torch.tensor(temp_txt15).to(device)
  txt16=torch.tensor(temp_txt16).to(device)
  print(txt.size(),img.size())
  # img = torch.cat(200*[img])
  # txt = torch.cat(200*[txt])

  # logits,z_new,output= self.forward(txt,img,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16)

  hidden_pred= model_pre_trained(txt,img)
  print(hidden_pred.size())
  # _,logit_offen,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = hm_model(txt,img,hidden_pred)
#logit_inten_fin = inten.detach().cpu().numpy()
  fin = torch.nn.Softmax(dim=1)(hidden_pred)
  preds = np.array([pred / pred.sum() for pred in fin])
  print (preds)