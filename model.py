#!/usr/bin/env python
# coding: utf-8

# # Assignment to build a Machine Learning model
# ### sample-data from the GitHub repo
# ### https://github.com/internbuddy/foster-app.git
# 
# 

# ### import required libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set()


# ### Load Data From xlsx File

# In[2]:


# Read in the data
path='Data_Science_2020_v2.xlsx'

data = pd.read_excel(path,
header=0,
index_col=False,
keep_default_na=True
)


# In[3]:


# View the top rows of the dataset
data.head(3)


# In[4]:


data.columns


# ### Stastical Description of Data

# In[5]:


data['Performance_PG']=data['Performance_PG'].transform(lambda x:x.fillna('No degree') )


# In[6]:


df1=data['Performance_PG']=data['Performance_PG'].str.split('/',expand=True)
df2=data['Performance_UG']=data['Performance_UG'].str.split('/',expand=True)
df3=data['Performance_12']=data['Performance_12'].str.split('/',expand=True)
df4=data['Performance_10']=data['Performance_10'].str.split('/',expand=True)


# In[7]:


df1.drop(1,axis=1,inplace=True)
df2.drop(1,axis=1,inplace=True)
df3.drop(1,axis=1,inplace=True)
df4.drop(1,axis=1,inplace=True)


# In[8]:


df1 = df1.rename(columns={0:'Performance_PG'})
df2 = df2.rename(columns={0:'Performance_UG'})
df3 = df3.rename(columns={0:'Performance_12'})
df4 = df4.rename(columns={0:'Performance_10'})


# In[9]:


score=pd.concat([df1,df2,df3,df4],axis=1)


# In[10]:


data.drop(['Performance_PG','Performance_UG','Performance_12','Performance_10'],axis=1,inplace=True)


# In[11]:


data=pd.concat([data,score],axis=1)


# In[12]:


df2=data['Performance_UG']=data['Performance_UG'].astype(float)
df3=data['Performance_12']=data['Performance_12'].astype(float)
df4=data['Performance_10']=data['Performance_10'].astype(float)


# In[13]:


df2=data['Performance_UG']=data['Performance_UG'].transform(lambda x:x.fillna(data['Performance_UG'].mean()))
df3=data['Performance_12']=data['Performance_12'].transform(lambda x:x.fillna(data['Performance_12'].mean()))
df4=data['Performance_10']=data['Performance_10'].transform(lambda x:x.fillna(data['Performance_10'].mean()))


# In[14]:



from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[15]:


std=StandardScaler()
lable=preprocessing.LabelEncoder()


# In[16]:


data['Other skills']=data['Other skills'].transform(lambda x:x.fillna('unknow'))
data['Degree']=data['Degree'].transform(lambda x:x.fillna('unknow'))
data['Stream']=data['Stream'].transform(lambda x:x.fillna('unknow'))


# In[17]:



Weightage_UG_2020=[]
c=0
for i in range(0,611):
    
    Year = data.iloc[i,9]
    UG = data.iloc[i,7]
    
    if (Year==2020 and (UG =='Bachelor of Engineering (B.E)' or UG =='Bachelor of Technology (B.Tech)')):
        Weightage_UG_2020.append(10) 
    else:
        Weightage_UG_2020.append(0)
    c=c+1    
 


# In[18]:


Weightage_UG_2019=[]
c=0
for i in range(0,611):
    
    Year = data.iloc[i,9]
    UG = data.iloc[i,7]
    
    if (Year==2019 and (UG =='Bachelor of Engineering (B.E)' or UG =='Bachelor of Technology (B.Tech)')):
        Weightage_UG_2019.append(8) 
    else:
        Weightage_UG_2019.append(0)
    c=c+1    


# In[19]:


Weightage_UG_2018=[]
c=0
for i in range(0,611):
    
    Year = data.iloc[i,9]
    UG = data.iloc[i,7]
    
    if (Year<=2018 and (UG =='Bachelor of Engineering (B.E)' or UG =='Bachelor of Technology (B.Tech)')):
        Weightage_UG_2018.append(5) 
    else:
        Weightage_UG_2018.append(0)
    c=c+1    


# In[20]:


Weightage_PG_2020=[]
c=0
for i in range(0,611):
    
    Year = data.iloc[i,9]
    PG = data.iloc[i,7]
    
    if (Year==2020 and (PG =='Master of Science (M.Sc)' or PG =='Master of Technology (M.Tech)')):
        Weightage_PG_2020.append(7) 
    else:
        Weightage_PG_2020.append(0)
    c=c+1    


    


# In[21]:


Weightage_PG_2019=[]
c=0
for i in range(0,611):
    
    Year = data.iloc[i,9]
    PG = data.iloc[i,7]
    
    if (Year<=2019 and (PG =='Master of Science (M.Sc)' or PG =='Master of Technology (M.Tech)')):
         Weightage_PG_2019.append(3) 
    else:
        Weightage_PG_2019.append(0)
    c=c+1    
 


# In[22]:


Weightage_Python3=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,2]==3):
        Weightage_Python3.append(10) 
    else:
        Weightage_Python3.append(0)
    c=c+1    


# In[23]:


Weightage_Python2=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,2]==2):
        Weightage_Python2.append(7) 
    else:
        Weightage_Python2.append(0)
    c=c+1    
 


# In[24]:


Weightage_Python1=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,2]==1):
        Weightage_Python1.append(3) 
    else:
        Weightage_Python1.append(0)
    c=c+1    


# In[25]:


Weightage_Rprog3=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,3]==3):
        Weightage_Rprog3.append(10) 
    else:
        Weightage_Rprog3.append(0)
    c=c+1    


# In[26]:


Weightage_Rprog2=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,3]==2):
        Weightage_Rprog2.append(7) 
    else:
        Weightage_Rprog2.append(0)
    c=c+1    


# In[27]:


Weightage_Rprog1=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,3]==1):
        Weightage_Rprog1.append(3) 
    else:
        Weightage_Rprog1.append(0)
    c=c+1    


# In[28]:


Weightage_ds3=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,4]==3):
        Weightage_ds3.append(10) 
    else:
        Weightage_ds3.append(0)
    c=c+1    


# In[29]:


Weightage_ds2=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,4]==2):
        Weightage_ds2.append(7) 
    else:
        Weightage_ds2.append(0)
    c=c+1    


# In[30]:


Weightage_ds1=[]
c=0
for i in range(0,611):
    
    if (data.iloc[i,4]==1):
        Weightage_ds1.append(3) 
    else:
        Weightage_ds1.append(0)
    c=c+1    


# In[31]:



Weightage_Oth_ML=[]
c=0


for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'Machine Learning' in line:
    
        Weightage_Oth_ML.append(3) 
    else:
        Weightage_Oth_ML.append(0)
    c=c+1    


# In[32]:


Weightage_Oth_DL=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'Deep Learning' in line:
    
        Weightage_Oth_DL.append(3) 
    else:
        Weightage_Oth_DL.append(0)
    c=c+1    


# In[33]:


Weightage_Oth_NLP=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'NLP' in line:
    
        Weightage_Oth_NLP.append(3) 
    else:
        Weightage_Oth_NLP.append(0)
    c=c+1    

 


# In[34]:


Weightage_Oth_SDA=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'Statistical Data Analysis' in line:
    
        Weightage_Oth_SDA.append(3) 
    else:
        Weightage_Oth_SDA.append(0)
    c=c+1    
Weightage_Oth_SDA=pd.Series(Weightage_Oth_SDA)


# In[35]:


Weightage_Oth_AWS=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'AWS' in line:
    
        Weightage_Oth_AWS.append(3) 
    else:
        Weightage_Oth_AWS.append(0)
    c=c+1    
Weightage_Oth_AWS=pd.Series(Weightage_Oth_AWS)


# In[36]:


Weightage_Oth_MYSQL=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'MySQL' in line:
    
        Weightage_Oth_MYSQL.append(3) 
    else:
        Weightage_Oth_MYSQL.append(0)
    c=c+1    
Weightage_Oth_MYSQL=pd.Series(Weightage_Oth_MYSQL)


# In[37]:


Weightage_Oth_NOSQL=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'NoSQL' in line:
    
        Weightage_Oth_NOSQL.append(3) 
    else:
        Weightage_Oth_NOSQL.append(0)
    c=c+1    


# In[38]:


Weightage_Oth_EXL=[]
c=0

for i in range(0,611):
    
   
    line=data.iloc[i,5]
    if 'Excel' in line:
    
        Weightage_Oth_EXL.append(3) 
    else:
        Weightage_Oth_EXL.append(0)
    c=c+1    
Weightage_Oth_EXL=pd.Series(Weightage_Oth_EXL)


# In[39]:


Weightage=pd.DataFrame(columns=['Weightage_Oth_EXL','Weightage_Oth_NOSQL','Weightage_Oth_MYSQL','Weightage_Oth_AWS','Weightage_Oth_SDA','Weightage_Oth_NLP','Weightage_Oth_DL','Weightage_Oth_ML','Weightage_ds1','Weightage_ds2','Weightage_ds3','Weightage_Rprog1','Weightage_Rprog2','Weightage_Rprog3','Weightage_Python1','Weightage_Python2','Weightage_Python3','Weightage_PG_2019','Weightage_PG_2020','Weightage_UG_2018','Weightage_UG_2019','Weightage_UG_2020'])


# In[40]:


Weightage['Weightage_Oth_EXL']=pd.Series(Weightage_Oth_EXL)
Weightage['Weightage_Oth_NOSQL']=pd.Series(Weightage_Oth_NOSQL)
Weightage['Weightage_Oth_MYSQL']=pd.Series(Weightage_Oth_MYSQL)
Weightage['Weightage_Oth_AWS']=pd.Series(Weightage_Oth_AWS)
Weightage['Weightage_Oth_SDA']=pd.Series(Weightage_Oth_SDA)
Weightage['Weightage_Oth_NLP']=pd.Series(Weightage_Oth_NLP)
Weightage['Weightage_Oth_DL']=pd.Series(Weightage_Oth_DL)
Weightage['Weightage_Oth_ML']=pd.Series(Weightage_Oth_ML)
Weightage['Weightage_ds1']=pd.Series(Weightage_ds1)
Weightage['Weightage_ds2']=pd.Series(Weightage_ds2)
Weightage['Weightage_ds3']=pd.Series(Weightage_ds3)
Weightage['Weightage_Rprog1']=pd.Series(Weightage_Rprog1)
Weightage['Weightage_Rprog2']=pd.Series(Weightage_Rprog2)
Weightage['Weightage_Rprog3']=pd.Series(Weightage_Rprog3)
Weightage['Weightage_Python1']=pd.Series(Weightage_Python1)
Weightage['Weightage_Python2']=pd.Series(Weightage_Python2) 
Weightage['Weightage_Python3']=pd.Series(Weightage_Python3) 
Weightage['Weightage_PG_2019']=pd.Series(Weightage_PG_2019)
Weightage['Weightage_PG_2020']=pd.Series(Weightage_PG_2020)
Weightage['Weightage_UG_2018']=pd.Series(Weightage_UG_2018)
Weightage['Weightage_UG_2019']=pd.Series(Weightage_UG_2019)
Weightage['Weightage_UG_2020']=pd.Series(Weightage_UG_2020)


# In[41]:


Weightage.head(20)


# ### The total Weightage of the Students

# In[42]:


total=Weightage.sum(axis = 1) 
total.head(20)


# In[43]:


Weightage['total']=total
    


# In[44]:


Weightage.head()


# In[45]:


data['total']=total


# In[46]:


Selection_list=[]


for i in range(0,611):
    line=data.iloc[i,14]
    if line>=40:
        Selection_list.append('Congratulation ! Your profile has been shortlisted for Data Scientist')
    
    else:
        Selection_list.append('Sorry ! Your profile did not qualify for further discussion')
    c=c+1    
Selection_list=pd.Series(Selection_list)
print(Selection_list)


# In[47]:


data['Selection_list']=pd.Series(Selection_list)


# In[48]:


data.head()


# In[49]:


data['Selection_list'].value_counts()


# In[50]:


plt.figure(figsize=(10,7))
chains=data['Selection_list'].value_counts()
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Slection list",size=20,pad=20)
plt.xlabel("Number of Students",size=15)


# In[51]:


data2=data[['Application_ID','Selection_list']]
data2.head(50)


# In[52]:


print('Application_ID','                       Result')
print('---------------------------------------------------------------------------------------')
for i in range(0,611):
    line=data2.iloc[i,1]
    if line=='Congratulation ! Your profile has been shortlisted for Data Scientist':
        print('{0},           {1}'.format(data2.iloc[i,0],data2.iloc[i,1]))
    


# # Details of shortlisted Candiadtes

# In[53]:


print('Application_ID','                       Result')
print('---------------------------------------------------------------------------------------')
for i in range(0,611):
    line=data2.iloc[i,1]
    if line=='Sorry ! Your profile did not qualify for further discussion':
        print('{0},           {1}'.format(data2.iloc[i,0],data2.iloc[i,1]))


# # Details of Non Qualified Candidates

# In[54]:


data_labled=data.apply(lable.fit_transform)


# In[55]:


data_labled


# In[56]:


data_scaled=std.fit_transform(data_labled)


# In[57]:


data_scaled=pd.DataFrame(data_scaled,columns=data.columns)


# In[58]:


data_scaled.head()


# ### Models for the given data

# In[59]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


# In[60]:


final_data=pd.DataFrame(data_labled)


# In[61]:


x =final_data.drop(['Application_ID','Selection_list','total','Institute','Degree','Stream'],axis=1)
y =data['Selection_list']


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[64]:


array = [LogisticRegressionCV(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=50,n_jobs=5),
        AdaBoostClassifier(),
        GradientBoostingClassifier()]


# In[65]:


for i in range (0,len(array)):
    array[i].fit(x_train,y_train)


# In[66]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report


# In[67]:


l=[]
for i in range (0,len(array)):
    y_pred=array[i].predict(x_test)
    print(y_pred)
    l.append(accuracy_score(y_pred,y_test))
print(l)


# In[68]:


final = pd.DataFrame(l,index=('Logistic regreesion','Decision tree','Randomforest','Ada boost','Gradient boost'),columns=['Accuracy score'])
final


# In[69]:


dec = DecisionTreeClassifier()


# In[70]:


rf_model = dec.fit(x_train,y_train)
rf_model


# In[71]:


y_pred = dec.predict(x_test)
print(y_pred)


# In[72]:


print(classification_report(y_pred,y_test))


# ### Both the precision and recall score are high, which implies our model able to predict and detect perfectly.
# ​

# In[73]:


# Saving model to disk
pickle.dump(dec, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,1,0,3,433,121,8,61,14,80]]))

