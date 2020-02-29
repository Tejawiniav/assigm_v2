#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

url = 'http://localhost:8000/predict_api'
r = requests.post(url,json={'Current City':5,'Python (out of 3)':1,'R Programming (out of 3)':0,'Data Science (out of 3)':0,'Other skills':453,'Institute':121,'Degree':8,'Stream':61,'Current Year Of Graduation':14,'Performance_PG':91,'Performance_UG':81,'Performance_12':90,'Performance_10':80,'total':30
})

print(r.json())

