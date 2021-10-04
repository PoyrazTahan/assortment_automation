# %%
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
from prep_tree import *
# %% Read sales and sku lkup
folder     = folder
file_sales = f'{folder}//sales//monthly_main_20210909_0232.h5'
data       = pd.read_hdf(file_sales,key='df')

file_lkup = f'{folder}//Sku_lkup//skin2_manuel_endcoing_20211004_1220.xlsx'
df_lkup   = pd.read_excel(file_lkup,sheet_name='Sheet1',dtype={'upc_13':str,'upc_full':str,'upc':str,'upc13_noc_check':str,'UPC 12 DIGIT':str,'EAN 14':str,'itemno':str,'vendor':str})
# %%
interested_columns = [
    'upc_13', 'upc', 'itemdesc', 'exclude',
        'CATEGORY_grp1', 'BENEFIT_grp1', 'SKIN_grp1', 'TYPE_grp1', 'BODY_grp1', 'DERMA_grp1', 'TIER_grp1',
        'brand','subbrand', 'vendor', 'mfrname', 'pl_type','itemstatus',
        'sku_SALES3', 'sku_UNITS3', 'sku_MAX_STORE3', 'sku_AVG_PRICE', 
]
df_lkup = df_lkup[interested_columns]

data['MONTH_YEAR'] = data['DATE'].dt.strftime('%Y-%m')

data['covid_exclude'] = 0
data.loc[data['MONTH_YEAR'] >= '2020-03','covid_exclude'] = 1

def non_brand_hypothesis():
    tier1 = [
        'OLAY', 'CERAVE', 'NEUTROGENA', 'LOREAL', 'ACNOMEL', 'CLEARASIL',
       'WAHL', 'COLLAGEN', 'BIO-OIL', 'ACNEFREE', 'DIFFERIN', 'ROC',
       'MASQUE BAR', 'NADS', 'FINISHING TOUCH', 'BURTS BEES',
       'ART NATURALS'
    ]
    tier2 = [
        'CETAPHIL', 'BIORE', 'AVEENO', 'NAIR', 'PONDS', 'GOLD BOND',
       'JOLEN', 'VEET', 'INSTRUMENTAL BEAUTY', 'AQUAPHOR',
       'FREEMAN BEAUTY', 'BEAUTY INFUSION', 'PANOXYL', 'CONAIR',
       'SALLY HANSEN', 'ST. IVES', 'COLONIAL DAMES', 'CLEAN & CLEAR',
       'SIMPLE', 'AMLACTIN', 'PORCELANA', 'GARNIER', 'IN.GREDIENTS',
       'HEMPZ', 'OKEEFFES', 'CAKE BEAUTY', 'MISS SPA', 'LUMENE'
    ]
    tier3 = [
        'STRIDEX', 'ALBOLENE', 'PALMERS', 'DICKINSONS', 'DOVE',
       'RITE AID BRAND', 'CUREL', 'JERGENS', 'THAYERS', 'SUAVE',
       'LUBRIDERM', 'LIVE CLEAN', 'EUCERIN', 'NIVEA', 'REAL TECHNIQUES',
       'QUEEN HELENE', 'SEA BREEZE', 'CARMEX', 'NOXZEMA', 'BAG BALM',
       'KERI', 'VASELINE', 'CHAPSTICK', 'SOFTLIPS', 'OXY', 'VANICREAM',
       'REBELS REFINERY', 'ABOUT FACE', 'UDDERLY SMOOTH', 'SHEA MOISTURE',
       'SKY ORGANICS', 'DR. TEALS', 'EOS', 'SUNBUM'
    ]
    ################################
    ################################
    brand_hypo_cat1 = {
        'ch1_brand_vs_NEUTROGENA_vs_CERAVE_vs_CETAPHIL_vs_OLAY_vs_RAD_vs_other': {
            'NEUTROGENA': ['NEUTROGENA'],
            'CERAVE'    : ['CERAVE'],
            'CETAPHIL'  : ['CETAPHIL'],
            'OLAY'      : ['OLAY'],
            'RAD'       : ['RITE AID BRAND'],
            'other'     : [e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','CERAVE','CETAPHIL','OLAY','RITE AID BRAND']],
        }
    }
    brand_hypo_cat2 = {
        'ch2_brand_vs_NEUTROGENA_other': {
            'NEUTROGENA': ['NEUTROGENA'],
            'other'     : [e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA']],
        }
    }
    brand_hypo_cat3 = {
        'ch3_brand_vs_NEUTROGENA_vs_OLAY_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY']],        
        }
    }
    brand_hypo_cat4 = {
        'ch4_brand_vs_NEUTROGENA_vs_CETAPHIL_vs_OLAY_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'CETAPHIL':['CETAPHIL'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','CETAPHIL']],     
        }
    }
    brand_hypo_cat5 = {
        'ch5_brand_vs_NEUTROGENA_vs_CETAPHIL_vs_OLAY_vs_RAD_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'CETAPHIL':['CETAPHIL'],
            'RAD':['RITE AID BRAND'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','CETAPHIL','RITE AID BRAND']],       
        }
    }
    brand_hypo_cat6 = {
        'ch6_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'AVEENO':['AVEENO'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','AVEENO']],        
        }
    }
    brand_hypo_cat7 = {
        'ch7_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_RAD_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'AVEENO':['AVEENO'],
            'RAD':['RITE AID BRAND'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','AVEENO','RITE AID BRAND']],        
        }
    }
    brand_hypo_cat8 = {
        'ch8_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_RAD_vs_CERAVE_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'AVEENO':['AVEENO'],
            'RAD':['RITE AID BRAND'],
            'CERAVE':['CERAVE'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','AVEENO','RITE AID BRAND','CERAVE']],        
        }
    }
    brand_hypo_cat9 = {
        'ch9_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_RAD_vs_CERAVE_vs_CETAPHIL_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'AVEENO':['AVEENO'],
            'RAD':['RITE AID BRAND'],
            'CERAVE':['CERAVE'],
            'CETAPHIL':['CETAPHIL'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','AVEENO','RITE AID BRAND','CERAVE','CETAPHIL']],        
        }
    }
    brand_hypo_cat10 = {
        'ch10_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_RAD_vs_CERAVE_vs_CETAPHIL_vs_LOREAL_vs_other': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'AVEENO':['AVEENO'],
            'RAD':['RITE AID BRAND'],
            'CERAVE':['CERAVE'],
            'CETAPHIL':['CETAPHIL'],
            'LOREAL':['LOREAL'],
            'other':[e for e in tier1+tier2+tier3 if e not in ['NEUTROGENA','OLAY','AVEENO','RITE AID BRAND','CERAVE','CETAPHIL','LOREAL']],        
        }
    }
    
    '''
    brand_hypo_cat1 = {
        'ch1_brand_vs_NEUTROGENA_vs_CERAVE_vs_CETAPHIL_vs_OLAY_vs_RAD_vs_premium_vs_mid_vs_value': {
            'NEUTROGENA':['NEUTROGENA'],
            'CERAVE':['CERAVE'],
            'CETAPHIL':['CETAPHIL'],
            'OLAY':['OLAY'],
            'RAD':['RITE AID BRAND'],
            'premium':[e for e in tier1 if e not in ['NEUTROGENA','CERAVE','CETAPHIL','OLAY','RITE AID BRAND']],
            'mid':[e for e in tier2 if e not in ['NEUTROGENA','CERAVE','CETAPHIL','OLAY','RITE AID BRAND']],
            'value':[e for e in tier3 if e not in ['NEUTROGENA','CERAVE','CETAPHIL','OLAY','RITE AID BRAND']],        
        }
    }
    brand_hypo_cat2 = {
        'ch2_brand_vs_NEUTROGENA_premium_vs_mid_vs_value': {
            'NEUTROGENA':['NEUTROGENA'],
            'premium':[e for e in tier1 if e not in ['NEUTROGENA']],
            'mid':[e for e in tier2 if e not in ['NEUTROGENA']],
            'value':[e for e in tier3 if e not in ['NEUTROGENA']],        
        }
    }
    brand_hypo_cat3 = {
        'ch3_brand_vs_NEUTROGENA_vs_OLAY_premium_vs_mid_vs_value': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'premium':[e for e in tier1 if e not in ['NEUTROGENA','OLAY']],
            'mid':[e for e in tier2 if e not in ['NEUTROGENA','OLAY']],
            'value':[e for e in tier3 if e not in ['NEUTROGENA','OLAY']],        
        }
    }
    brand_hypo_cat4 = {
        'ch4_brand_vs_NEUTROGENA_vs_CETAPHIL_vs_OLAY_vs_premium_vs_mid_vs_value': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'CETAPHIL':['CETAPHIL'],
            'premium':[e for e in tier1 if e not in ['NEUTROGENA','OLAY','CETAPHIL']],
            'mid':[e for e in tier2 if e not in ['NEUTROGENA','OLAY','CETAPHIL']],
            'value':[e for e in tier3 if e not in ['NEUTROGENA','OLAY','CETAPHIL']],        
        }
    }
    brand_hypo_cat5 = {
        'ch5_brand_vs_NEUTROGENA_vs_CETAPHIL_vs_OLAY_vs_RAD_vs_premium_vs_mid_vs_value': {
            'NEUTROGENA':['NEUTROGENA'],
            'OLAY':['OLAY'],
            'CETAPHIL':['CETAPHIL'],
            'RAD':['RITE AID BRAND'],
            'premium':[e for e in tier1 if e not in ['NEUTROGENA','OLAY','CETAPHIL','RITE AID BRAND']],
            'mid':[e for e in tier2 if e not in ['NEUTROGENA','OLAY','CETAPHIL','RITE AID BRAND']],
            'value':[e for e in tier3 if e not in ['NEUTROGENA','OLAY','CETAPHIL','RITE AID BRAND']],        
        }
    }
    '''
    hypothesis = []
    hypothesis = hypothesis + [brand_hypo_cat1,brand_hypo_cat2,brand_hypo_cat3,brand_hypo_cat4,brand_hypo_cat5,
                               brand_hypo_cat6,brand_hypo_cat7,brand_hypo_cat8,brand_hypo_cat9,brand_hypo_cat10]
    return hypothesis
# %% Name the necessary columns

main_column_dict_rad = {
    'id'           : 'upc_13',
    'time'         : 'MONTH_YEAR',
    'value'        : 'SALES',
    'unit'         : 'UNITS',
    'weights'      : 'UNIQUE_STORE_SCAN',
    'tail'         : 'exclude',
    'last_n_months': 'covid_exclude',
    'brand'        : 'brand',
    'status'       : 'itemstatus',
    }
additional_column_dict = {
    # Additioanl keep columns
    'own_brands'  : 'pl_type',
    'venodr'      : 'vendor',
    'clinet_sku'  : 'upc',
    'manufacturer': 'mfrname'
}
# %% Create the main data frame
cols = list( main_column_dict_rad.values() )
cols = [e for e in cols if e not in ('exclude','brand','itemstatus')]
data = data[cols].copy()
df = data.merge(df_lkup, on=main_column_dict_rad['id'],how='inner')
df['upc_13'] = 'upc_' + df['upc_13'].astype(str)

my_brands = df[df.pl_type != 'NONE'].brand.unique()
# %% LVL 0
root = Tree(df, node_name = 'skin', my_brands=my_brands, main_column_dict = main_column_dict_rad, additional_column_dict = additional_column_dict)
# %% LVL 1
root.add_hypothesis(addition= non_brand_hypothesis() )
# %%
root_chld = root.create_children(hypo='ch10_brand_vs_NEUTROGENA_vs_OLAY_vs_AVEENO_vs_RAD_vs_CERAVE_vs_CETAPHIL_vs_LOREAL_vs_other')
#%% LVL 2
neutrogena  = root_chld[0] # NEUTROGENA
olay        = root_chld[1] # OLAY
aveeno      = root_chld[2] # AVEENO
rad         = root_chld[3] # RAD
cerave      = root_chld[4] # CERAVE
cetaphil    = root_chld[5] # CETAPHIL
loreal      = root_chld[6] # LOREAL
brand_other = root_chld[7] # other

# %%
neutrogena_chld       = neutrogena.create_children(hypo='h7_CATEGORY_grp1_vs_MAKEUP_REMOVER_c1_p35_vs_CARE_c1_p34_vs_CLEANSER_c2_p30')
olay_chld             = olay.create_children(hypo='h7_CATEGORY_grp1_vs_CARE_c1_p81_vs_CLEANSER_c1_p11_vs_MAKEUP_REMOVER_c2_p6')
cetaphil_chld         = cetaphil.create_children(hypo='h1_BODY_grp1_vs_FACIAL_c1_p89_vs_HAND_BODY_c1_p10')
rad_chld              = rad.create_children(hypo='h13_CATEGORY_grp1_vs_MAKEUP_REMOVER_c1_p25_vs_CLEANSER_c1_p17_vs_CARE_c2_p56')
brand_other_chld_chld = brand_other_chld = brand_other.create_children(hypo='h1_DERMA_grp1_vs_NONE_c1_p59_vs_DERMA_c1_p40')
# %% #########################
neut_mur = neutrogena_chld[0]
neut_care = neutrogena_chld[1]
neut_cln = neutrogena_chld[2]

olay_care     = olay_chld[0]
olay_clean    = olay_chld[1]
olay_mur_exfo = olay_chld[2]

cetaphil_faical = cetaphil_chld[0]
cetaphil_body = cetaphil_chld[1]

rad_mur = rad_chld[0]
rad_clean = rad_chld[1]
rad_care = rad_chld[2]

brand_other_nonderma = brand_other_chld[0] # non derme
brand_other_derma = brand_other_chld[1] # derma
# %%
neut_mur.create_children(hypo='h1_BENEFIT_grp1_vs_CLEAN_c1_p76_vs_MOIST_FEEL_c1_p23')
neut_care.create_children(hypo='h1_BODY_grp1_vs_FACIAL_c1_p90_vs_HAND_BODY_c1_p9')
neut_cln.create_children(hypo='h1_BENEFIT_grp1_vs_ACNE_c1_p53_vs_MOIST_FEEL_c3_p46')

olay_care_chld = olay_care.create_children(hypo = 'h1_BENEFIT_grp1_vs_ANTI_AGE_c1_p50_vs_MOIST_FEEL_c1_p49') # h1_BODY_grp1_vs_FACIAL_c1_p93_vs_HAND_BODY_c1_p6
olay_clean_chld = olay_clean.create_children(hypo = 'h5_SKIN_grp1_vs_OILY_c2_p50_vs_ALL_c2_p49') # h5_SKIN_grp1_vs_OILY_c2_p50_vs_ALL_c2_p49
olay_mur_exfo_chld = olay_mur_exfo.create_children(hypo = 'h3_SKIN_grp1_vs_MIXED_c1_p24_vs_SENSETIVE_c3_p75')

cetaphil_faical_chld =cetaphil_faical.create_children(hypo='h2_CATEGORY_grp1_vs_CARE_c1_p41_vs_CLEANSER_c2_p58')
cetaphil_body_chld =cetaphil_body.create_children(hypo='h4_SKIN_grp1_vs_OILY_c1_p51_vs_DRY_c1_p28_vs_ALL_c1_p20')
rad_mur_chld = rad_mur.create_children(hypo='h2_BENEFIT_grp1_vs_CLEAN_c2_p94_vs_ACNE_c1_p5')
rad_clean_chld = rad_clean.create_children(hypo='h7_BENEFIT_grp1_vs_MOIST_FEEL_c1_p11_vs_CLEAN_c3_p88')
rad_care_chld = rad_care.create_children(hypo='h1_BODY_grp1_vs_FACIAL_c1_p67_vs_HAND_BODY_c1_p32')
brand_other_derma_chld = brand_other_derma.create_children(hypo='h4_TIER_grp1_vs_mid_c1_p50_vs_value_c1_p33_vs_premium_c1_p15')
brand_other_nonderma_chld = brand_other_nonderma.create_children(hypo='h26_CATEGORY_grp1_vs_CARE_c1_p54_vs_CLEANSER_c1_p25_vs_DEPILATORIES_c1_p12_vs_EXFOLIATOR_c2_p8')
# %%
neut_mur_clean = neut_mur.get_child('g1p76c1_nCLEAN')
neut_mur_feel = neut_mur.get_child('g2p23c1_nMOIST_FEEL')
neut_care_facial = neut_care.get_child('g1p90c1_nFACIAL')
neut_care_body = neut_care.get_child('g2p9c1_nHAND_BODY')
neu_cln_acne = neut_cln.get_child('g1p53c1_nACNE')
neu_cln_moist = neut_cln.get_child('g2p46c3_nMOIST_FEEL')

# %%
neut_mur_clean_chld = neut_mur_clean.create_children(hypo='h1_SKIN_grp1_vs_ALL_c1_p91_vs_OILY_c2_p8')
neut_mur_feel.create_children(hypo='h1_SKIN_grp1_vs_ALL_c1_p69_vs_OILY_c2_p30')
neut_care_facial_chld = neut_care_facial.create_children(hypo='h2_BENEFIT_grp1_vs_MOIST_FEEL_c2_p85_vs_ANTI_AGE_c1_p14')
neut_care_body.create_children(hypo='h1_SKIN_grp1_vs_SENSETIVE_c1_p76_vs_DRY_c1_p23')
neu_cln_acne_chld = neu_cln_acne.create_children(hypo='h3_SKIN_grp1_vs_ALL_c1_p38_vs_OILY_c3_p61')
neu_cln_moist_chld = neu_cln_moist.create_children(hypo='h1_SKIN_grp1_vs_ALL_c1_p46_vs_SENSETIVE_c3_p53')
# %%
cetaphil_faical_care = cetaphil_faical_chld[0] # h11_SKIN_grp1_vs_ALL_c2_p78_vs_MIXED_c1_p15_vs_SENSETIVE_c1_p5
cetaphil_faical_clean = cetaphil_faical_chld[1] # h4_BENEFIT_grp1_vs_MOIST_FEEL_c1_p66_vs_CLEAN_c1_p27_vs_ANTI_AGE_c1_p5

cetaphil_body_oily = cetaphil_body_chld[0]
cetaphil_body_dry = cetaphil_body_chld[1]
cetaphil_body_all = cetaphil_body_chld[2]

# %%
neut_mur_clean_all = neut_mur_clean_chld[0] # stop
neut_care_facial_moist = neut_care_facial_chld[0] # h1_TYPE_grp1_vs_CREAM_c1_p58_vs_GEL_c5_p41
neut_care_facial_moist.create_automatic_hypothesis(children_to_focus=['TYPE_grp1'])
neu_cln_acne_all = neu_cln_acne_chld[0] # stop
neu_cln_acne_oily = neu_cln_acne_chld[1] # h37_TYPE_grp1_vs_CREAM_c1_p12_vs_LQ_and_SERUM_c2_p32_vs_GEL_c2_p54
neu_cln_moist_sens = neu_cln_moist_chld[1] # stop

olay_care_age = olay_care_chld[0] # h1_subbrand_vs_OLAY REGENERIST_c1_p79_vs_OLAY TOTAL EFFECTS_c4_p20
olay_care_moist = olay_care_chld[1] # h2_SKIN_grp1_vs_ALL_c2_p85_vs_MIXED_c2_p14

rad_mur_clean = rad_mur_chld[0] # 'g1p94c2_nCLEAN' - h4_SKIN_grp1_vs_ALL_c1_p86_vs_SENSETIVE_c1_p6_vs_OILY_c1_p6
rad_mur_acne = rad_mur_chld[1] # 'g2p5c1_nACNE' - stop
rad_clean_moist = rad_clean_chld[0] # 'g1p11c1_nMOIST_FEEL' - stop
rad_clean_clean = rad_clean_chld[1] # 'g2p88c3_nCLEAN' - h1_SKIN_grp1_vs_ALL_c1_p93_vs_SENSETIVE_c1_p6
rad_care_facial = rad_care_chld[0] # 'g1p67c1_nFACIAL' - h9_BENEFIT_grp1_vs_ACNE_c1_p36_vs_ANTI_AGE_c1_p22_vs_MOIST_FEEL_c2_p40
rad_care_body = rad_care_chld[1] # 'g2p32c1_nHAND_BODY' - stop

brand_other_derma_mid = brand_other_derma_chld[0] # 'h1_BODY_grp1_vs_HAND_BODY_c1_p81_vs_FACIAL_c1_p18'
brand_other_derma_value = brand_other_derma_chld[1] # 'h1_BODY_grp1_vs_HAND_BODY_c1_p82_vs_FACIAL_c1_p17'
brand_other_derma_prem = brand_other_derma_chld[2] # 'h1_BENEFIT_grp1_vs_MOIST_FEEL_c1_p64_vs_ANTI_AGE_c1_p35'

brand_other_nonderma_care = brand_other_nonderma_chld[0] #  'h4_TIER_grp1_vs_value_c1_p52_vs_premium_c1_p29_vs_mid_c1_p18'
brand_other_nonderma_cln = brand_other_nonderma_chld[1] # 'h2_TIER_grp1_vs_mid_c2_p83_vs_value_c1_p16'
brand_other_nonderma_depi = brand_other_nonderma_chld[2] # 'h2_TIER_grp1_vs_mid_c2_p92_vs_premium_c1_p7'
brand_other_nonderma_exfo = brand_other_nonderma_chld[3] # 'h1_BENEFIT_grp1_vs_ACNE_c1_p55_vs_CLEAN_c4_p44'
# %%
neut_care_facial_moist.create_children('h1_TYPE_grp1_vs_CREAM_c1_p58_vs_GEL_c5_p41')
neu_cln_acne_oily.create_children('h37_TYPE_grp1_vs_CREAM_c1_p12_vs_LQ_and_SERUM_c2_p32_vs_GEL_c2_p54')

olay_care_age_chld = olay_care_age.create_children('h1_subbrand_vs_OLAY REGENERIST_c1_p79_vs_OLAY TOTAL EFFECTS_c4_p20')
olay_care_moist_chld = olay_care_moist.create_children('h2_SKIN_grp1_vs_ALL_c2_p85_vs_MIXED_c2_p14')

_ = cetaphil_faical_care.create_children(hypo='h11_SKIN_grp1_vs_ALL_c2_p78_vs_MIXED_c1_p15_vs_SENSETIVE_c1_p5')
cetaphil_faical_clean_chld = cetaphil_faical_clean.create_children(hypo='h4_BENEFIT_grp1_vs_MOIST_FEEL_c1_p66_vs_CLEAN_c1_p27_vs_ANTI_AGE_c1_p5')


_ = rad_mur_clean.create_children(hypo = 'h4_SKIN_grp1_vs_ALL_c1_p86_vs_SENSETIVE_c1_p6_vs_OILY_c1_p6')
_ = rad_clean_clean.create_children(hypo = 'h1_SKIN_grp1_vs_ALL_c1_p93_vs_SENSETIVE_c1_p6')
_ = rad_care_facial.create_children(hypo = 'h9_BENEFIT_grp1_vs_ACNE_c1_p36_vs_ANTI_AGE_c1_p22_vs_MOIST_FEEL_c2_p40')

brand_other_derma_mid_chld = brand_other_derma_mid.create_children(hypo='h1_BODY_grp1_vs_HAND_BODY_c1_p81_vs_FACIAL_c1_p18')
brand_other_derma_value_chld = brand_other_derma_value.create_children(hypo='h1_BODY_grp1_vs_HAND_BODY_c1_p82_vs_FACIAL_c1_p17')
brand_other_derma_prem_chld = brand_other_derma_prem.create_children(hypo='h1_BENEFIT_grp1_vs_MOIST_FEEL_c1_p63_vs_ANTI_AGE_c1_p36')

_ = brand_other_nonderma_care.create_children(hypo='h4_TIER_grp1_vs_value_c1_p52_vs_premium_c1_p29_vs_mid_c1_p18' )
_ = brand_other_nonderma_cln.create_children(hypo='h3_TIER_grp1_vs_value_c1_p19_vs_mid_c2_p80' ) # 
_ = brand_other_nonderma_depi.create_children(hypo='h2_TIER_grp1_vs_mid_c2_p92_vs_premium_c1_p7' )
_ = brand_other_nonderma_exfo.create_children(hypo='h18_BENEFIT_grp1_vs_ACNE_c1_p46_vs_CLEAN_c2_p28_vs_ANTI_AGE_c2_p25' )
# %%
olay_care_age_regen = olay_care_age_chld[0] # h1_TYPE_grp1_vs_CREAM_c1_p82_vs_LQ_and_SERUM_c1_p17
olay_care_age_ao = olay_care_age_chld[1]

cetaphil_faical_clean_moist = cetaphil_faical_clean_chld[0] # h1_SKIN_grp1_vs_OILY_c1_p33_vs_SENSETIVE_c3_p66
cetaphil_faical_clean_clean = cetaphil_faical_clean_chld[1] # h2_TYPE_grp1_vs_BAR_c2_p78_vs_CREAM_c1_p21
cetaphil_faical_clean_age = cetaphil_faical_clean_chld[2] # stop

brand_other_derma_mid_body = brand_other_derma_mid_chld[0] # 'h1_BENEFIT_grp1_vs_MOIST_FEEL_c1_p48_vs_HEAL_c3_p51'
brand_other_derma_mid_facial = brand_other_derma_mid_chld[1] # 'h4_CATEGORY_grp1_vs_CARE_c1_p50_vs_CLEANSER_c1_p33_vs_EXFOLIATOR_c1_p16'
brand_other_derma_value_body = brand_other_derma_value_chld[0] # 'h3_BENEFIT_grp1_vs_MOIST_FEEL_c1_p42_vs_HEAL_c2_p57'
brand_other_derma_value_facial = brand_other_derma_value_chld[1] # 'h4_BENEFIT_grp1_vs_HEAL_c1_p42_vs_MOIST_FEEL_c1_p39_vs_ANTI_AGE_c1_p18'
brand_other_derma_prem_moist = brand_other_derma_prem_chld[0] # 'h2_TYPE_grp1_vs_LQ_and_SERUM_c1_p15_vs_CREAM_c3_p84'
brand_other_derma_prem_age = brand_other_derma_prem_chld[1] # 'h2_SKIN_grp1_vs_ALL_c2_p72_vs_DRY_c1_p27'
# %%
olay_care_age_regen.create_children(hypo='h1_TYPE_grp1_vs_CREAM_c1_p82_vs_LQ_and_SERUM_c1_p17')

_ = cetaphil_faical_clean_moist.create_children(hypo='h1_SKIN_grp1_vs_OILY_c1_p33_vs_SENSETIVE_c3_p66')
_ = cetaphil_faical_clean_clean.create_children(hypo='h2_TYPE_grp1_vs_BAR_c2_p78_vs_CREAM_c1_p21')


brand_other_derma_mid_body_cld = brand_other_derma_mid_body.create_children(hypo='h1_BENEFIT_grp1_vs_MOIST_FEEL_c1_p48_vs_HEAL_c3_p51')
brand_other_derma_mid_facial_cld = brand_other_derma_mid_facial.create_children(hypo='h4_CATEGORY_grp1_vs_CARE_c1_p50_vs_CLEANSER_c1_p33_vs_EXFOLIATOR_c1_p16')
brand_other_derma_value_body_cld = brand_other_derma_value_body.create_children(hypo='h3_BENEFIT_grp1_vs_MOIST_FEEL_c1_p42_vs_HEAL_c2_p57')
brand_other_derma_value_facial_cld = brand_other_derma_value_facial.create_children(hypo='h4_BENEFIT_grp1_vs_HEAL_c1_p42_vs_MOIST_FEEL_c1_p39_vs_ANTI_AGE_c1_p18')
brand_other_derma_prem_moist_cld = brand_other_derma_prem_moist.create_children(hypo='h2_TYPE_grp1_vs_LQ_and_SERUM_c1_p15_vs_CREAM_c3_p84')
brand_other_derma_prem_age_cld = brand_other_derma_prem_age.create_children(hypo='h2_SKIN_grp1_vs_ALL_c2_p72_vs_DRY_c1_p27')


# %%
root.draw_tree(p_type = 'segment_focus')
#%%

brand_other_nonderma = brand_other.get_child('g1p57c1_nNONE')
brand_other_derma = brand_other.get_child('g2p42c1_nDERMA')
# %%
# %%
