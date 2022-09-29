#!/usr/bin/env python
# coding: utf-8

# # Case 2 - Team 12

# * Joey van Alphen
# * Mohamed Garad
# * Nusret Kaya
# * Shereen Macnack

# # 1. Data inladen

# ### Dataset 1: Maandcijfers Nederlandse luchthavens van nationaal belang

# In[1]:


#pip install cbsodata


# In[2]:


import cbsodata
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm


# In[3]:


data1 = pd.DataFrame(cbsodata.get_data('37478hvv'))


# In[4]:


#data1.shape


# In[5]:


data1.info(verbose=True, show_counts=True)


# In[6]:


data1.head()


# ### Dataset 2: Emissies naar lucht door de Nederlandse economie; nationale rekeningen

# In[7]:


data2 = pd.DataFrame(cbsodata.get_data('83300NED'))


# In[8]:


# data2.shape


# In[9]:


data2.info()


# In[10]:


data2.head()


# # 2. Data filteren

# ### 1. Aviation data filteren

# In[11]:


aviation_data = data1[['ID', 'Luchthavens', 'Perioden', 'Overlandbewegingen_1', 'Terreinbewegingen_2','TotaalAlleVluchten_3', 'TotaalVertrokkenVluchten_9', 'TotaalAantalPassagiers_12','EuropaTotaal_22','EULanden_54','OverigEuropa_55', 'Afrika_57','Amerika_63', 'Azie_67', 'Oceanie_71', 'TotaalGoederenvervoer_43', 'TotalePostvervoer_74']]


# In[ ]:





# In[12]:


#aviation_data die is gefilterd op alleen het totaal van alle luchhavens van nationaal belang
alle_luchthavens = aviation_data[aviation_data['Luchthavens']=='Totaal luchthavens van nationaal belang']


# In[13]:


alle_luchthavens.head(50)


# In[14]:


# alle_luchthavens filteren op volledige jaren ipv maanden voor totaal 
value_list = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
boolean_series = alle_luchthavens.Perioden.isin(value_list)
alle_luchthavens = alle_luchthavens[boolean_series]
alle_luchthavens_index = alle_luchthavens.reset_index(drop = True)
alle_luchthavens_index.head(50)


# In[ ]:





# In[15]:


#aviation_data die de individuele luchthavens bevat
individuele_luchthavens = aviation_data[aviation_data['Luchthavens']!='Totaal luchthavens van nationaal belang']
individuele_luchthavens_index = individuele_luchthavens.reset_index(drop = True)


# In[ ]:





# In[16]:


#individuele_luchthavens_index gefilterd op volledige jaren ipv maanden voor individuele luchthavens
value_list = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011','2012', '2013', '2014', '2015', '2016','2017', '2018', '2019','2020']
boolean_series = individuele_luchthavens_index.Perioden.isin(value_list)
individuele_luchthavens_index = individuele_luchthavens_index[boolean_series]
individuele_luchthavens_index.head()


# ### 2. Emissies data filteren

# In[17]:


data2.head()


# In[18]:


co2_emissies = data2[['ID','NederlandseEconomie','Perioden', 'CO2_1']]


# In[19]:


co2_emissies_luchtvaart = co2_emissies[co2_emissies['NederlandseEconomie']=='51 Vervoer door de lucht']


# In[20]:


co2_emissies_luchtvaart.head(50)


# In[21]:


#Filteren vanaf 1997 om de andere dataset te matchen
co2_emissies_luchtvaart = co2_emissies_luchtvaart[co2_emissies_luchtvaart['Perioden']>='1997'].reset_index(drop=True)
co2_emissies_luchtvaart = co2_emissies_luchtvaart.drop(['ID'], axis=1) 


# In[22]:


co2_emissies_luchtvaart.head(50)


# In[23]:


co2_emissies_luchtvaart.columns = ['Emissie categorie', 'Perioden', 'CO2 uitstoot (mln kg)']
co2_emissies_luchtvaart.head(30)


# In[24]:


# Dataframes combineren
samengestelde_tabel = alle_luchthavens_index.merge(co2_emissies_luchtvaart, on='Perioden', how='left')

samengestelde_tabel = samengestelde_tabel.drop(['ID', 'Emissie categorie'], axis = 1)


# In[25]:


# Kolom namen veranderen van samengestelde tabel
samengestelde_tabel = samengestelde_tabel.rename ({'Perioden': 'Jaar', 'Overlandbewegingen_1':'Overlandbewegingen' , 'Terreinbewegingen_2':'Terreinbewegingen', 'TotaalAlleVluchten_3': 'Totaal aantal vluchten', 'TotaalVertrokkenVluchten_9':'Totaal vertrokken vluchten','TotaalAantalPassagiers_12': 'Totaal aantal passagiers', 'EuropaTotaal_22':'Europa totaal','EULanden_54': 'EU landen', 'OverigEuropa_55':'Overig Europa', 'Afrika_57':'Afrika','Amerika_63': 'Amerika','Azie_67': 'Azie', 'Oceanie_71':'Oceanie', 'TotaalGoederenvervoer_43':'Totaal goederenvervoer','TotalePostvervoer_74':'Totaal postvervoer', 'CO2_1': 'CO2 emissies in jaar'}, axis = 1)
samengestelde_tabel.head(30)


# In[ ]:





# In[26]:


fig = px.scatter(samengestelde_tabel, x='Totaal aantal vluchten', y ='CO2 uitstoot (mln kg)', trendline="ols", )


# In[27]:


fig1 = px.histogram(individuele_luchthavens_index, x='Perioden', y='TotaalAlleVluchten_3', color = 'Luchthavens')


# In[28]:


fig2 = px.line(samengestelde_tabel, x='Jaar', y='CO2 uitstoot (mln kg)')


# # 3. Maken van de Streamlit app

# In[29]:


header = st.container()


# In[30]:


with header:
    st.title('Aviation data blog')
    st.markdown('In deze blog gebruiken we data van het CBS over de maandelijkse cijfers van Nederlandse luchthavens. Wij willen inzicht krijgen in welke mate deze cijfers gerelateerd zijn tot de jaarlijkse uitstoot van de luchtvaart industrie in Nederland met behulp van een tweede dataset. Daarmee gaan we kijken of we een eenvoudige voorspellende formule kunnen maken.')


# In[31]:


aviation_data_streamlit_table = aviation_data[['Luchthavens', 'Perioden', 'TotaalAlleVluchten_3', 'TotaalAantalPassagiers_12','TotaalGoederenvervoer_43', 'TotalePostvervoer_74']]
aviation_data_streamlit_table.columns = ['Luchthavens', 'Perioden', 'Totaal aantal vluchten', 'Totaal aantal passagiers', 'Totale goederenvervoer', 'Totale postvervoer']
value_list = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011','2012', '2013', '2014', '2015', '2016','2017', '2018', '2019','2020']
boolean_series = aviation_data_streamlit_table.Perioden.isin(value_list)
aviation_data_streamlit_table = aviation_data_streamlit_table[boolean_series]
aviation_data_streamlit_table.head(30)


# In[32]:


st.header('Data inladen met API')
st.markdown("Beide datasets zijn ingeladen door eerst de package **cbsodata** van het CBS te downloaden. Vervolgens hebben we de data binnen kunnen halen met een pandas dataframe en door de **get_data()** aan te roepen met de tabelcode van het CBS. *Voorbeeld: df = pd.DataFrame(cbsodata.get_data('...'))* ")


# In[33]:


st.subheader('Dataset 1 met de maandelijkse cijfers van Nederlandse luchthavens')
st.markdown("Deze eerste dataset bevat meer dan honderd kolommen die alle vluchten onderverdeelt in onder andere de bestemmingen, vetrokken en aangekomen passagiers, geregelde niet-geregelde vluchten etc. Voor ons voorspellende formule kijken we alleen naar het totaal van de belangrijkste variabelen:passagiers en goederen- en postvervoer. Om straks de tweede dataset te kunnen koppelen hebben we deze tabel alleen gefilterd op het totaal aantal van alle luchthavens.")


# In[34]:


InputAirport = st.sidebar.selectbox("Select Airport", ("Totaal luchthavens van nationaal belang", "Amsterdam Airport Schiphol", "Rotterdam The Hague Airport", "Eindhoven Airport", "Maastricht Aachen Airport", "Groningen Airport Eelde"))


# In[35]:


AirportSelect = aviation_data_streamlit_table[aviation_data_streamlit_table["Luchthavens"] == InputAirport]


# In[36]:


st.dataframe(AirportSelect)


# In[37]:


st.markdown("Met behulp van deze visualisaties kunnen we de 3 variabelen per luchthaven vergelijken.")


# In[38]:


fig1 = px.histogram(individuele_luchthavens_index, x='Luchthavens', y='TotaalAlleVluchten_3', color = 'Luchthavens', animation_frame = 'Perioden', animation_group = 'Luchthavens' )
fig2 = px.histogram(individuele_luchthavens_index, x='Luchthavens', y='TotaalGoederenvervoer_43', color = 'Luchthavens', animation_frame = 'Perioden', animation_group = 'Luchthavens' )
fig3 = px.histogram(individuele_luchthavens_index, x='Luchthavens', y='TotalePostvervoer_74', color = 'Luchthavens', animation_frame = 'Perioden', animation_group = 'Luchthavens' )


# In[39]:


st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)


# ### tweede tabel

# In[40]:


st.subheader('Dataset 2 met de emissies')
st.markdown("Deze dataset bevat bevat cijfers over emissies van schadelijke stoffen naar lucht voor zover die samenhangen met Nederlandse economische activiteiten. We hebben deze gefilterd op de emissies van de categorie 'Vervoer door de lucht' en de bijbehorende CO2 uitstoot ")


# In[41]:


st.dataframe(co2_emissies_luchtvaart)


# In[42]:


st.subheader("De samengestelde dataset")
st.markdown('Met behulp van de **.merge()** methode en een **Left join** hebben we nu één tabel. We zien echter dat de eerste 6 jaar van de variabele '"Totale post vervoer missen"'. Deze vullen we op met het gemiddelde van deze hele kolom met **.fillna()**. Ook voegen we 4 kolommen toe: Totale uitstoot sinds meting, vulgraad_pax, vulgraad_vracht, vulgraad_post')


# In[43]:


aviation_data_streamlit_table = aviation_data_streamlit_table[aviation_data_streamlit_table['Luchthavens']=='Totaal luchthavens van nationaal belang']
samengestelde_tabel_streamlit = aviation_data_streamlit_table.merge(co2_emissies_luchtvaart, on='Perioden', how='left')
samengestelde_tabel_streamlit = samengestelde_tabel_streamlit.drop(['Emissie categorie'], axis = 1)


# In[44]:


samengestelde_tabel_streamlit = samengestelde_tabel_streamlit.fillna(value=samengestelde_tabel_streamlit['Totale postvervoer'].mean())


# In[45]:


samengestelde_tabel_streamlit['Totale uitstoot sinds meting'] = samengestelde_tabel_streamlit['CO2 uitstoot (mln kg)'].cumsum()
samengestelde_tabel_streamlit['vulgraad_pax'] = samengestelde_tabel_streamlit['Totaal aantal passagiers'] / samengestelde_tabel_streamlit['Totaal aantal vluchten']
samengestelde_tabel_streamlit['vulgraad_vracht'] = samengestelde_tabel_streamlit['Totale goederenvervoer'] / samengestelde_tabel_streamlit['Totaal aantal vluchten']
samengestelde_tabel_streamlit['vulgraad_post'] = samengestelde_tabel_streamlit['Totale goederenvervoer'] / samengestelde_tabel_streamlit['Totaal aantal vluchten']


# In[46]:


samengestelde_tabel_streamlit.head()


# In[47]:


st.dataframe(samengestelde_tabel_streamlit)


# ## Plots

# In[48]:


st.header("Visualisaties")
st.markdown("Met behulp van visualisaties gaan we nu bepalen wat de correlatie is tussen verschillende variabelen en de totale uitstoot van CO2. ")
st.markdown("Als eerst bekijken we de CO2 uitstoot door de jaren heen. Hieruit valt direct op dat in tijden van crisis (2008 - 2009), maar vooral de coronacrisis (2020) de uitstoot heeft doen verlagen.")


# In[49]:


fig4 = px.line(samengestelde_tabel_streamlit, x='Perioden', y='CO2 uitstoot (mln kg)', title = 'CO2 emissie verloop')  
fig.update_xaxes(title_text='Jaren')


# In[50]:


st.plotly_chart(fig4)


# In[51]:


fig5 = go.Figure(data=[go.Scatter(
    x=samengestelde_tabel_streamlit['Perioden'],
    y=samengestelde_tabel_streamlit['CO2 uitstoot (mln kg)'],
    mode='markers',)
])
  
# Add dropdown
fig5.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["type", "scatter"],
                    label="Scatter Plot",
                    method="restyle"               
                ),
                dict(
                    args=["type", "bar"],
                    label="Bar Chart",
                    method="restyle"
                )
            ]),
            direction="down",
        ),
    ]
)

fig5.update_xaxes(title_text = 'Jaar')
fig5.update_yaxes(title_text = 'CO2 emissie (mln kg)')


# In[52]:


st.plotly_chart(fig5)


# In[53]:


st.markdown("Er lijk hier inderdaad sprake van een correlatie. We checken dit met een correlatie matrix")


# In[54]:


corr = np.corrcoef(samengestelde_tabel_streamlit['Totaal aantal vluchten'], samengestelde_tabel_streamlit['CO2 uitstoot (mln kg)'])

fig_c, ax = plt.subplots()
fig = sns.heatmap(corr, ax = ax, annot = True)

st.write(fig_c)


# In[55]:


st.markdown("De correlatie coefficient van 0,77 laat zien dat de realtie tussen deze twee variabelen inderdaad sterk is.")


# In[56]:


st.markdown("We bekijken nogmaals de relatie tussen het totale aantal vluchten in een jaar tegenover de CO2 uitstoot in dat jaar.")


# In[57]:


fig6 = px.scatter(samengestelde_tabel_streamlit, x='Totaal aantal vluchten', y ='CO2 uitstoot (mln kg)', trendline="ols", title = 'Verband tussen CO2 uitstoot en het totaal aantal vluchten')


# In[58]:


st.plotly_chart(fig6)


# In[59]:


st.markdown("De regressielijn geeft de formule: **CO2 uitstoot (mln kg) = 0,0142941*Totaal aantal vluchten + 4917,18**. We kunnen hier dus mee de CO2 uitstoot van een jaar voorspellen op basis van het aantal verwachte vluchten in dat jaar")


# In[60]:


st.header("Andere variabelen")


# In[61]:


fig7 = px.scatter(samengestelde_tabel_streamlit, y='CO2 uitstoot (mln kg)', x ='vulgraad_pax', title = 'Verband tussen vulgraad van passagiers en de CO2 uitstoot')
fig7.update_xaxes(title_text='Vulgraad passagiers')


# In[62]:


st.plotly_chart(fig7)


# In[63]:


fig8 = px.scatter(samengestelde_tabel_streamlit, y='CO2 uitstoot (mln kg)', x ='vulgraad_vracht', title = 'Verband tussen vulgraad van vracht en de CO2 uitstoot')
fig8.update_xaxes(title_text='Vulgraad vracht')


# In[64]:


st.plotly_chart(fig8)


# In[65]:


fig9 = px.scatter(samengestelde_tabel_streamlit, y='CO2 uitstoot (mln kg)', x ='vulgraad_post', title = 'Verband tussen vulgraad van post en de CO2 uitstoot')
fig9.update_xaxes(title_text='Vulgraad post')


# In[66]:


st.plotly_chart(fig9)


# In[67]:


fig10 = px.line(samengestelde_tabel_streamlit, x='Perioden', y = 'Totale uitstoot sinds meting', title = 'Totale CO2 emmissies sinds meting')
fig10.update_xaxes(title_text='Jaar')



# In[68]:


st.plotly_chart(fig10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




