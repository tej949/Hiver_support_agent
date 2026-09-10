from pathlib import Path
import pandas as pd

p=Path('data/processed/golden_v2/golden_set_locked.csv')
df=pd.read_csv(p,dtype=str,low_memory=False).fillna('')
if len(df)!=200: raise ValueError(f'Expected 200 rows, got {len(df)}')

# AMZ id: intent | action | handling | evidence
s='''
1 delivery|investigation|ESCALATE|YES
2 delivery|investigation|ESCALATE|YES
3 delivery|investigation|ESCALATE|YES
4 delivery|investigation|ESCALATE|YES
5 delivery|phone_chat|ESCALATE|YES
6 return_refund_replacement|refund_replacement|ESCALATE|YES
7 order_status_tracking|phone_chat|ESCALATE|YES
8 delivery|investigation|ESCALATE|YES
9 delivery|investigation|ESCALATE|YES
10 delivery|investigation|ESCALATE|YES
11 delivery|investigation|ESCALATE|YES
12 delivery|investigation|ESCALATE|YES
13 delivery|refund_replacement|ESCALATE|YES
14 order_status_tracking|information|AUTO-HANDLE|YES
15 delivery|phone_chat|ESCALATE|YES
16 delivery|investigation|AUTO-HANDLE|YES
17 delivery|investigation|ESCALATE|YES
18 delivery|phone_chat|ESCALATE|NO
19 delivery|information|AUTO-HANDLE|YES
20 delivery|phone_chat|ESCALATE|YES
21 delivery|phone_chat|ESCALATE|NO
22 delivery|investigation|AUTO-HANDLE|YES
23 delivery|investigation|ESCALATE|YES
24 order_status_tracking|information|AUTO-HANDLE|YES
25 order_status_tracking|information|AUTO-HANDLE|YES
26 order_status_tracking|information|AUTO-HANDLE|YES
27 information_general|acknowledgement|AUTO-HANDLE|YES
28 order_status_tracking|information|AUTO-HANDLE|YES
29 customer_service_complaint|investigation|ESCALATE|YES
30 delivery|investigation|ESCALATE|YES
31 information_general|information|AUTO-HANDLE|YES
32 delivery|information|ESCALATE|YES
33 customer_service_complaint|phone_chat|ESCALATE|YES
34 delivery|phone_chat|ESCALATE|YES
35 delivery|information|AUTO-HANDLE|YES
36 order_status_tracking|information|AUTO-HANDLE|YES
37 delivery|investigation|ESCALATE|YES
38 delivery|information|AUTO-HANDLE|YES
39 information_general|acknowledgement|AUTO-HANDLE|YES
40 order_change_cancel|information|AUTO-HANDLE|YES
41 payment_billing|refund_replacement|ESCALATE|YES
42 return_refund_replacement|refund_replacement|ESCALATE|YES
43 return_refund_replacement|acknowledgement|AUTO-HANDLE|YES
44 return_refund_replacement|refund_replacement|ESCALATE|YES
45 delivery|refund_replacement|ESCALATE|YES
46 return_refund_replacement|refund_replacement|ESCALATE|YES
47 return_refund_replacement|information|AUTO-HANDLE|YES
48 return_refund_replacement|phone_chat|ESCALATE|YES
49 return_refund_replacement|phone_chat|ESCALATE|YES
50 return_refund_replacement|refund_replacement|AUTO-HANDLE|YES
51 payment_billing|phone_chat|ESCALATE|NO
52 return_refund_replacement|investigation|ESCALATE|NO
53 return_refund_replacement|refund_replacement|ESCALATE|NO
54 return_refund_replacement|refund_replacement|ESCALATE|NO
55 return_refund_replacement|investigation|ESCALATE|NO
56 return_refund_replacement|refund_replacement|ESCALATE|YES
57 payment_billing|information|AUTO-HANDLE|YES
58 return_refund_replacement|refund_replacement|ESCALATE|NO
59 return_refund_replacement|refund_replacement|ESCALATE|NO
60 delivery|refund_replacement|ESCALATE|NO
61 return_refund_replacement|investigation|ESCALATE|NO
62 return_refund_replacement|refund_replacement|ESCALATE|YES
63 delivery|investigation|ESCALATE|YES
64 payment_billing|information|AUTO-HANDLE|YES
65 delivery|information|AUTO-HANDLE|YES
66 shipping_delivery_options|information|AUTO-HANDLE|YES
67 shipping_delivery_options|information|AUTO-HANDLE|YES
68 return_refund_replacement|refund_replacement|ESCALATE|YES
69 return_refund_replacement|information|AUTO-HANDLE|YES
70 order_status_tracking|information|AUTO-HANDLE|YES
71 order_status_tracking|information|AUTO-HANDLE|YES
72 payment_billing|information|AUTO-HANDLE|NO
73 payment_billing|refund_replacement|AUTO-HANDLE|YES
74 shipping_delivery_options|investigation|ESCALATE|NO
75 payment_billing|investigation|ESCALATE|YES
76 shipping_delivery_options|phone_chat|ESCALATE|NO
77 shipping_delivery_options|phone_chat|ESCALATE|NO
78 payment_billing|investigation|ESCALATE|YES
79 account_access|phone_chat|ESCALATE|YES
80 product_device_issue|investigation|ESCALATE|NO
81 account_access|phone_chat|ESCALATE|YES
82 account_access|information|AUTO-HANDLE|YES
83 account_access|phone_chat|ESCALATE|YES
84 payment_billing|information|AUTO-HANDLE|YES
85 account_access|investigation|ESCALATE|YES
86 account_access|privacy_restriction|ESCALATE|YES
87 account_access|investigation|ESCALATE|NO
88 account_access|phone_chat|ESCALATE|YES
89 account_access|information|AUTO-HANDLE|YES
90 account_access|phone_chat|ESCALATE|NO
91 account_access|information|AUTO-HANDLE|YES
92 payment_billing|investigation|ESCALATE|NO
93 account_access|information|AUTO-HANDLE|YES
94 account_access|information|AUTO-HANDLE|YES
95 shipping_delivery_options|information|AUTO-HANDLE|YES
96 product_device_issue|acknowledgement|AUTO-HANDLE|YES
97 product_device_issue|information|AUTO-HANDLE|YES
98 product_device_issue|information|AUTO-HANDLE|YES
99 product_device_issue|information|AUTO-HANDLE|YES
100 product_device_issue|investigation|ESCALATE|YES
101 product_device_issue|information|AUTO-HANDLE|YES
102 product_device_issue|information|AUTO-HANDLE|YES
103 product_device_issue|phone_chat|ESCALATE|YES
104 information_general|acknowledgement|AUTO-HANDLE|YES
105 product_device_issue|phone_chat|ESCALATE|NO
106 payment_billing|investigation|ESCALATE|NO
107 product_device_issue|information|AUTO-HANDLE|YES
108 product_device_issue|phone_chat|ESCALATE|YES
109 product_device_issue|information|AUTO-HANDLE|YES
110 product_device_issue|information|AUTO-HANDLE|YES
111 information_general|information|AUTO-HANDLE|YES
112 product_device_issue|information|AUTO-HANDLE|YES
113 product_device_issue|self_service|AUTO-HANDLE|YES
114 return_refund_replacement|acknowledgement|AUTO-HANDLE|YES
115 product_device_issue|self_service|AUTO-HANDLE|YES
116 product_device_issue|information|AUTO-HANDLE|YES
117 product_device_issue|information|AUTO-HANDLE|NO
118 delivery|acknowledgement|AUTO-HANDLE|YES
119 shipping_delivery_options|information|AUTO-HANDLE|YES
120 shipping_delivery_options|information|AUTO-HANDLE|YES
121 shipping_delivery_options|information|AUTO-HANDLE|YES
122 shipping_delivery_options|information|AUTO-HANDLE|NO
123 delivery|investigation|ESCALATE|YES
124 shipping_delivery_options|information|AUTO-HANDLE|NO
125 shipping_delivery_options|information|AUTO-HANDLE|NO
126 shipping_delivery_options|phone_chat|ESCALATE|YES
127 shipping_delivery_options|information|AUTO-HANDLE|YES
128 shipping_delivery_options|information|AUTO-HANDLE|YES
129 shipping_delivery_options|acknowledgement|AUTO-HANDLE|YES
130 shipping_delivery_options|information|AUTO-HANDLE|YES
131 shipping_delivery_options|investigation|ESCALATE|YES
132 shipping_delivery_options|information|AUTO-HANDLE|YES
133 customer_service_complaint|investigation|ESCALATE|YES
134 customer_service_complaint|phone_chat|ESCALATE|YES
135 customer_service_complaint|investigation|ESCALATE|YES
136 customer_service_complaint|phone_chat|ESCALATE|YES
137 customer_service_complaint|investigation|ESCALATE|YES
138 customer_service_complaint|acknowledgement|AUTO-HANDLE|YES
139 customer_service_complaint|phone_chat|ESCALATE|YES
140 customer_service_complaint|investigation|ESCALATE|YES
141 customer_service_complaint|investigation|ESCALATE|YES
142 customer_service_complaint|information|AUTO-HANDLE|NO
143 customer_service_complaint|investigation|ESCALATE|YES
144 customer_service_complaint|phone_chat|ESCALATE|NO
145 customer_service_complaint|investigation|ESCALATE|NO
146 customer_service_complaint|information|AUTO-HANDLE|NO
147 customer_service_complaint|phone_chat|ESCALATE|YES
148 customer_service_complaint|phone_chat|ESCALATE|YES
149 customer_service_complaint|investigation|ESCALATE|NO
150 customer_service_complaint|information|AUTO-HANDLE|NO
151 delivery|investigation|ESCALATE|YES
152 shipping_delivery_options|information|AUTO-HANDLE|YES
153 product_device_issue|information|AUTO-HANDLE|YES
154 information_general|information|AUTO-HANDLE|YES
155 shipping_delivery_options|investigation|ESCALATE|YES
156 delivery|investigation|ESCALATE|YES
157 product_device_issue|phone_chat|ESCALATE|YES
158 shipping_delivery_options|information|AUTO-HANDLE|YES
159 account_access|phone_chat|ESCALATE|YES
160 information_general|information|AUTO-HANDLE|YES
161 account_access|phone_chat|ESCALATE|YES
162 delivery|information|AUTO-HANDLE|YES
163 delivery|acknowledgement|AUTO-HANDLE|YES
164 order_change_cancel|information|AUTO-HANDLE|YES
165 order_change_cancel|information|AUTO-HANDLE|YES
166 delivery|investigation|ESCALATE|YES
167 order_change_cancel|information|AUTO-HANDLE|YES
168 order_change_cancel|investigation|ESCALATE|NO
169 order_change_cancel|phone_chat|ESCALATE|YES
170 order_change_cancel|investigation|ESCALATE|NO
171 account_access|self_service|AUTO-HANDLE|YES
172 account_access|information|ESCALATE|YES
173 payment_billing|investigation|ESCALATE|NO
174 payment_billing|investigation|ESCALATE|YES
175 product_device_issue|self_service|AUTO-HANDLE|YES
176 delivery|information|AUTO-HANDLE|YES
177 information_general|information|AUTO-HANDLE|YES
178 product_device_issue|investigation|ESCALATE|NO
179 customer_service_complaint|phone_chat|ESCALATE|YES
180 customer_service_complaint|information|AUTO-HANDLE|YES
181 customer_service_complaint|information|AUTO-HANDLE|YES
182 payment_billing|self_service|AUTO-HANDLE|YES
183 return_refund_replacement|acknowledgement|AUTO-HANDLE|YES
184 account_access|investigation|ESCALATE|NO
185 account_access|investigation|ESCALATE|YES
186 account_access|phone_chat|ESCALATE|YES
187 account_access|phone_chat|ESCALATE|YES
188 account_access|investigation|ESCALATE|YES
189 payment_billing|information|ESCALATE|YES
190 delivery|investigation|ESCALATE|YES
191 account_access|phone_chat|ESCALATE|YES
192 delivery|investigation|ESCALATE|YES
193 customer_service_complaint|phone_chat|ESCALATE|YES
194 delivery|investigation|ESCALATE|YES
195 product_device_issue|phone_chat|ESCALATE|YES
196 customer_service_complaint|phone_chat|ESCALATE|YES
197 payment_billing|investigation|ESCALATE|YES
198 customer_service_complaint|phone_chat|ESCALATE|YES
199 customer_service_complaint|investigation|ESCALATE|YES
200 delivery|phone_chat|ESCALATE|YES
'''

rows={}
for line in s.strip().splitlines():
    i,rest=line.split(' ',1); rows[int(i)]=rest.split('|')
if set(rows)!=set(range(1,201)): raise ValueError('Label mapping does not contain exactly AMZ-0001..0200')

reasons={
'delivery':'The customer has a delivery-related issue requiring the selected resolution.',
'order_status_tracking':'The customer is asking about order status or tracking.',
'order_change_cancel':'The customer needs help with an order change or cancellation.',
'return_refund_replacement':'The customer needs help with a return, refund, or replacement.',
'payment_billing':'The customer has a payment, billing, charge, or cashback issue.',
'account_access':'The customer has an account access or account-state issue.',
'product_device_issue':'The customer has a product or device issue.',
'shipping_delivery_options':'The customer has a shipping or delivery-options issue.',
'customer_service_complaint':'The customer reports an unresolved support/service issue.',
'information_general':'The customer is requesting general information or acknowledging a resolution.'}

def mkreason(intent,action,handling):
    x=reasons[intent]
    return x + (' The current case should be routed for human handling.' if handling=='ESCALATE' else ' The request appears suitable for the selected automated handling.')

for i in range(1,201):
    gid=f'AMZ-{i:04d}'; idx=df.index[df.golden_id.astype(str)==gid].tolist()
    if len(idx)!=1: raise ValueError(f'{gid}: expected one row, found {len(idx)}')
    j=idx[0]; intent,action,handling,evidence=rows[i]
    df.loc[j,'gold_intent']=intent; df.loc[j,'gold_action']=action; df.loc[j,'gold_handling']=handling
    df.loc[j,'gold_reason']=mkreason(intent,action,handling); df.loc[j,'gold_evidence_sufficient']=evidence

df.to_csv(p,index=False,encoding='utf-8')

check=pd.read_csv(p,dtype=str,low_memory=False)
cols=['gold_intent','gold_action','gold_handling','gold_reason','gold_evidence_sufficient']
print('Restored labels.')
print('Rows:',len(check))
print('\nMissing labels:'); print(check[cols].isna().sum().to_string())
print('\nHandling:'); print(check.gold_handling.value_counts().to_string())
print('\nIntent:'); print(check.gold_intent.value_counts().to_string())
