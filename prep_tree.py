# %%
import pandas as pd
import numpy as np

# %%
folder = f'..//data'
""""
Regular Pipeline:
    Preliminary:
        - Endoce attributes on sku level
        - Prepare timeseries data
        - Create a folder structure // folders needed: data -> Hypo, Sku_lkup, vtm, model_input 

    -  1: Tree() : initialize
    -  2: possible_children : check the main columns
    -  3: get_possible_child_items() : checking which columns to foucs 
    -  4: create_automatic_hypothesis() : try to get combinations
    -  5: add_hypothesis() : create custom hypothesis
    -  6: calculate_hypothesis_distribtions() : checking and reduce # hypothesis
    -  7: save_model() : save the output to use at periscope
    -  8: create_children()
    -  9: list_children()
    - 10: get_child() and repeat from step 2 till end condition
    - 11: .draw_tree(p_type='structure') / .draw_tree(p_type='check') # powerpoint output
"""
class Tree:
    def __init__(self, purchase_data, node_name = 'root', main_column_dict = None, additional_column_dict = None,my_brands= [], time_period='M',children = None,path=[]):
        self.name = node_name
        self.children = []
        self.main_columns = main_column_dict
        self.additional_columns = additional_column_dict
        
        if children is not None:
            for child in children:
                self.add_child(child)

        self.data = purchase_data
        if not path: # if path is empty
            self.data = self.create_time_label(time_period=time_period)

            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler(feature_range=(0.0001, 1))
            scale = scaler.fit_transform(self.data[main_column_dict['weights']].values.reshape(-1,1))
            self.data['scale_weights'] = scale

            self.main_columns['scale_weights'] = 'scale_weights'

        self.object_created_columns = ['year_label_focus', 'year_label_all','datetime']
        self.possible_children = list( set(self.data.columns.tolist())-set(self.main_columns.values()) - set(self.additional_columns.values()) -set(self.object_created_columns))

        self.hypothesises = []
        self.chosen_hypothesis_name = None

        self.model_input_csv = None
        self.hypothesis_evaluate_sheet = None
        self.vtm_input_csv = None

        self.our_brands = my_brands
        self.path = path
        
        self.total_market_size = self.calculate_segment_size(node = self, non_tail=False, year_label='last_year1')

        self.__initilize_hypothesis()
        
    class hypothesis:
        def __init__(self, name, lookup_dict, break_attribute):
            self.name = name
            self.lookup_dict = lookup_dict
            self.break_attribute = break_attribute


    def __initilize_hypothesis(self, max_initial_label = 5):
        easy_focus_group = [possible_child for possible_child in self.possible_children if (self.get_possible_child_items(possible_child).shape[0] <= max_initial_label)&(not self.get_possible_child_items(possible_child).index.is_numeric())]
        
        if len(easy_focus_group) == 0:
            print(f'There is no easy breaks for {self.name} // Please choose focus group')
        elif len(easy_focus_group) > 0:
            self.create_automatic_hypothesis(children_to_focus=easy_focus_group, addtive=False)

    def clear_hypothesis(self):
        self.hypothesises = []

    def list_hypothesis(self):
        print(self.hypothesises)

    def create_time_label(self,time_period):
        self.data['datetime'] = pd.to_datetime(self.data[self.main_columns['time']])

        focus_period = self.get_focus_data()
        df_time_focus = self.group_time(focus_period, time_period=time_period, period_type='focus')

        all_period = self.data.copy()
        df_time_all = self.group_time(all_period, time_period=time_period, period_type='all')

        df_w_time_info = self.data.merge(df_time_all, on='datetime', how='left').merge(df_time_focus, on='datetime', how='left')
        return df_w_time_info

    def group_time(self, df, time_period, period_type='focus'):
        '''
        Groups the time into categories of years starting from the last time point depending on the frequency given
        time_period: M: monthly, Q:quarterly
        period_type: all: for printing, focus: model prep 
        '''
        time = df[self.main_columns['time']].unique()
        time.sort()
        time_index = pd.DatetimeIndex(time)

        divider = 12 if time_period == 'M' else 4 if time_period == 'Q' else None # determines the length of the year
        n_whole_years = len(time_index) // divider

        df_time = pd.DataFrame(index= time_index, columns=[f'year_label_{period_type}'])
        for year in range(n_whole_years):
            for label_length in range(divider):
                df_time.iloc[-(divider*(year)+label_length+1), 0] = f'last_year{year+1}'

        return df_time.fillna(f'last_year{year+2}').reset_index().rename(columns={'index':'datetime'})

    def get_possible_child_items(self, child_name, year_label='last_year1'):
        df = self.get_focus_data()

        return self.__calc_distribution(df, agg_col= child_name, eval_col = self.main_columns['value'], year_label=year_label)
    
    def add_child(self, data, node_name):
        path_to_add = self.path.copy()
        path_to_add.append(self)
        self.children.append(
            Tree(
                data, node_name,
                main_column_dict=self.main_columns,
                additional_column_dict=self.additional_columns,
                my_brands=self.our_brands,
                path=path_to_add
            )
        )

    def prepare_model_input_csv(self):
        df = self.prepare_model_input()
        # df = df_all.loc[(df_all[self.main_columns['tail']]==0)&(df_all[self.main_columns['last_n_months']]==1),:]

        l_col = list( self.main_columns.values() )
        hypo_cols = list(df.filter(regex='h\d*_(.+?)_vs').columns)

        df_m = df[l_col+hypo_cols]
        df_m = self.prepare_periscope_csv(df_m)      
        self.model_input_csv = df_m

        return df_m

    def prepare_periscope_csv(self,df):
        df = df.copy()

        df.columns = df.columns.str.replace(' ','_')
        df.columns = df.columns.str.replace('-','_')
        df.columns = df.columns.str.replace('[()]','',regex=True)
        df.loc[:,self.main_columns['time']] = self.periscope_date_format_conversion(df[self.main_columns['time']])

        df = df[(df[[self.main_columns['value'],self.main_columns['unit'],self.main_columns['scale_weights'] ]] > 0).all(1)].copy()
        return df

    def periscope_date_format_conversion(self,series_time):
        return pd.to_datetime(series_time, format='%Y-%m').dt.strftime('%m/%d/%Y')

    def prepare_evaluate_sheet(self):
        df_hypo = self.prepare_model_input() 

        for label in sorted(df_hypo['year_label_focus'].unique()):
            df_hypo[f'value_{label}'] = df_hypo[self.main_columns['value']].copy()
            df_hypo.loc[df_hypo['year_label_focus'] != label, f'value_{label}'] = np.NaN

            df_hypo[f'unit_{label}'] = df_hypo[self.main_columns['unit']].copy()
            df_hypo.loc[df_hypo['year_label_focus'] != label, f'unit_{label}'] = np.NaN

            df_hypo[f'wt_{label}'] = df_hypo[self.main_columns['scale_weights']].copy()
            df_hypo.loc[df_hypo['year_label_focus'] != label, f'wt_{label}'] = np.NaN

        self.hypothesis_evaluate_sheet = df_hypo
        return df_hypo


    def create_hypothesis_column(self, df, child, column_map):
        return df[child].map(column_map)

    def calculate_hypo_eval_metrics(self): #  TODO: this function to eliminate backcasting excel step, 
        # TODO: Find a way to get the output of the backcasting model
        # TODO: Find a way to calculate value growths
        df = self.hypothesis_evaluate_sheet
        hypo_cols = list(df.filter(regex='h\d*_(.+?)_vs').columns)
        other_cols = ['BRAND', 'VARIANT (SUB-BRAND)',
            'COLOUR OF COSMETIC', 'PRODUCT CLASS', 'PRODUCT FORM', 'SUB CATEGORY',
            'FORMAT', 'Max_Month_Year','Allergic', 'LONG LASTING','WATERPROFF'] # TODO make this derived using toher columns

        growth = self.calculate_eval_growths(df, hypo_cols)
        corr = self.calculate_eval_corr(df, hypo_cols, other_cols)

        return growth, corr

    def calculate_eval_growths(self, df, hypothesis_columns):
        value = [self.main_columns['value'] + '_sku']
        df = df[value + hypothesis_columns + ['DUMMY_SKU','LDESC','Max_Month_Year','value_y1_sku','value_y2_sku','unit_y1_sku','unit_y2_sku','wt_y1_sku','wt_y2_sku']].drop_duplicates()
        df['value_gr'] = df.value_y2_sku/df.value_y1_sku -1
        df['unit_gr'] = df.unit_y2_sku/df.unit_y1_sku -1
        df['wt_gr'] = df.wt_y2_sku/df.wt_y1_sku -1

        df['velocity_gr'] = (df.value_y2_sku/df.value_y1_sku) / (df.wt_y2_sku/df.wt_y1_sku) -1

        hypothesis_growths = {}
        for hypothesis in hypothesis_columns:
            hypothesis_growths[hypothesis] = df.groupby(hypothesis).agg({
                value[0]:'sum',
                'value_gr':'mean',
                'wt_gr':'mean',
                'velocity_gr':'mean'
                })

        return hypothesis_growths

    def calculate_eval_corr(self, df, x_axis_col, y_axis_col):
        from itertools import product

        corr_vals = {}
        for x, y in product(x_axis_col, y_axis_col):
            intersections = df.groupby([x,y]).agg({self.main_columns['value']:'sum'})
            corr_as_perc = intersections.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
            corr_vals[(x,y)] = corr_as_perc.unstack()

        return corr_vals

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def add_hypothesis(self, addition):
        """input of a dict or list of dict"""
        if isinstance(addition, dict): 
            self.hypothesises.append(addition)
        elif isinstance(addition, list):
            for hypo in addition:
                self.hypothesises.append(hypo)

            print(f'--{len(addition)} has been added! Total hypo: {len(self.hypothesises)}')

    def get_focus_data(self):
        return self.data[(self.data[self.main_columns['tail']]==0)&(self.data[self.main_columns['last_n_months']]==0)].copy()

    def calculate_hypothesis_distribtions(self,year_label='last_year1'):
        df = self.prepare_model_input()
        df = df.fillna("uncat")

        hypo_cols = list(df.filter(regex='h\d*_(.+?)_vs').columns)

        distributions = {f'hypothesis{idx}_dist':self.__calc_distribution(df, hypo, self.main_columns['value'],year_label=year_label) for idx, hypo in enumerate(hypo_cols) }

        return distributions

    def __calc_distribution(self, df, agg_col, eval_col,year_label):
        df = df.loc[df['year_label_focus'] == year_label, :]

        df = df.groupby(agg_col).agg({eval_col:'sum'})
        df['%'] = df[eval_col] / df[eval_col].sum()
        return df.sort_values('%',ascending=False) 


    def __invert_dictionary_of_lists(self, dict_to_reverse):
        inverted_dict = {}
        for k,v in dict_to_reverse.items():
            for x in v:
                inverted_dict.setdefault(x,k)
        
        return inverted_dict

    def prepare_model_input(self):
        import re

        df = self.get_focus_data()

        for hypo in self.hypothesises:
            key = list(hypo.keys())[0]
            child_col = re.search('h\d*_(.+?)_vs', key).group(1)
            col_map = self.__invert_dictionary_of_lists(hypo[key])
            df.loc[:,key] = self.create_hypothesis_column(df, child_col, col_map)

        return df

    def create_model_input_output(self):
        df_input = self.prepare_model_input_csv()
        df_input.loc[:,'Uniform_lvl1'] = 'dummy1'
        df_input.loc[:,'Uniform_lvl2'] = 'dummy2'
        df_hypo = self.prepare_evaluate_sheet()

        return df_input, df_hypo

    def get_parent_path_name(self):
        return [parent.name for parent in self.path]

    def save_model(self):
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d_%H%M")
        
        parent_name = self.get_parent_path_name()
        file_evaluate_excel = f'/Hypo/{now}_{"_".join(parent_name)}_{self.name}_output.xlsx'
        file_input_csv = f'/model_input/{now}_{"_".join(parent_name)}_{self.name}_model_input.csv'

        df_input, df_hypo = self.create_model_input_output()

        self.check_model_input()

        df_input.to_csv(folder+file_input_csv,index=False)
        df_hypo.to_excel(folder+file_evaluate_excel,index=False)


    def check_model_input(self): # TODO
        pass

    def create_children(self, hypo = 0):
        import re

        self.children = []
        self.chosen_hypothesis_name = hypo

        if isinstance(hypo, str):
            child_col = re.search('h\d*_(.+?)_vs', hypo).group(1)
            for h in self.hypothesises:
                if list(h.keys())[0] == hypo:
                    categories = list(h.values())[0]

        elif isinstance(hypo, int):
            hypothesis_dict = self.hypothesises[hypo]
            child_col = list(hypothesis_dict.keys())[0]
        
            categories = hypothesis_dict[child_col]

        self.children = [] # initialize a new children set

        children_data = self.prepare_children_data(child_col, categories)

        for child_name in categories:
            self.add_child(data=children_data[child_name], node_name=child_name)

        return self.children

    def prepare_children_data(self, child_column, categories = {}):
        children_dataframes = {}
        for key in categories:
            children_dataframes[key] = self.data[self.data[child_column].isin(categories[key])]

        return children_dataframes
    
    def list_children(self):
        children_names = [child.name for child in self.children]

        return children_names

    def get_child(self, child_name):
        child = [child for child in self.children if child.name == child_name]
        return child[0] 
            
    def save_tree(self, path):
        import pickle
        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d_%H%M")
        with open(path+f'/{now}_tree.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    def create_end_node_print(self,last_break,year_label):
        from tabulate import tabulate
        
        df_summary = self.data[self.data['year_label_all'] == year_label].groupby(last_break).agg({self.main_columns['value']:'sum'})
        df_summary.loc[:,self.main_columns['value']] = round(df_summary[self.main_columns['value']]/1000,2)
        df_summary = df_summary.sort_values(self.main_columns['value'],ascending=False).head(3)

        return tabulate(df_summary, headers='keys', tablefmt='psql')

    
    def __incremant_number(self, string, increment=1):
        def getNumbers(str):
            import re
            array = re.findall(r'[0-9]+', str)
            return array

        try:
            numbers_list = getNumbers(string)
            if len(numbers_list) != 1:
                raise ValueError('year_label should have only integer') 
        except (ValueError, IndexError):
            exit('Could not complete request.')

        number= int(numbers_list[0])
        return string.replace(str(number), str(number + increment))

    def draw_tree(self,p_type='segment_focus',non_tail=True,info_pad=50,level=0,year_label='last_year1',prior=''):
        """
        ptype: 
            brand_focus: $ size and owned brand share, 
            check: overall $ Share and group $ share,
            segment_focus:  overall $ Share and segment growth,
            structure: bare strcuture
        """
        
        prior_year = self.__incremant_number(year_label,1)

        segment_total = self.calculate_segment_size(node= self, non_tail = non_tail, year_label=year_label)
        segment_total_year_prior = self.calculate_segment_size(node= self, non_tail = non_tail, year_label=prior_year) # TODO: draw tree complete RAD view

        segment_growth = self.__calculate_growth(number_old = segment_total_year_prior, number_new = segment_total)
        brand_size = self.calculate_brand_size(node = self, non_tail = non_tail, year_label=year_label) 
        brand_perc = brand_size / segment_total
        sku_cnt, active_sku = self.__sku_count(node = self, non_tail = non_tail, year_label=year_label)

        if not self.path:
            direct_parent = root = self
        else:
            root = self.path[0]
            direct_parent = self.path[-1]

        total_market_size = self.calculate_segment_size(root, non_tail=non_tail, year_label=year_label)
        parent_market_size = self.calculate_segment_size(direct_parent, non_tail=non_tail, year_label=year_label)

        node_percent = segment_total/total_market_size
        parent_perc = segment_total/parent_market_size
        
        indent='-'+'--'*level+str(level)+':'
        info = f'\n-{indent}{self.name}:'

        structure_len = len(info)
        padding = ' ' * (info_pad - structure_len)

        info += padding
        if p_type == 'brand_focus':
            info += f'{round(segment_total/1000000,2)}M USD /  {round(brand_perc*100,2)}%'
        elif p_type == 'check':
            info += f'total: {round(node_percent*100,0)}% / group: {round(parent_perc*100,0)}%'
        elif p_type == 'segment_focus':
            info += f'{round(segment_total/1000000,2)}M USD / sku_cnt: {sku_cnt} actv: {active_sku} / total_share: {round(node_percent*100,0)}% / growth: {round(segment_growth*100,0)}%'
        elif p_type == 'structure':
            pass

        level_str = prior + info

        for child in self.children:
            level_str = child.draw_tree(level=level+1,year_label=year_label,prior=level_str,p_type=p_type)

        if level==0:
            print(level_str)
        else:
            return level_str

    def print_end_node(self,level=0,last_break='BRAND',year_label='last_year1'):
        indent='-'+'--'*level+str(level)+':'

        if len(self.list_children())==0:
            end_node = self.create_end_node_print(last_break=last_break,year_label=year_label)
            path_line = self.get_node_path()
            print(f'-{indent}{path_line}:\n' + end_node)

        for child in self.children:
            child.print_end_node(level=level+1,last_break=last_break,year_label=year_label)


    def __calculate_growth(self, number_old, number_new):
        return (number_new - number_old) / number_old

    def __sku_count(self, node, year_label='last_year1', non_tail = False):
        df = node.data.loc[node.data[node.main_columns['tail']]==0].copy() if non_tail else node.data.copy()
        df = df[df['year_label_all'] == year_label].copy()

        active_data = df[df[self.main_columns['status']] == 'ACTV' ]
        return df[self.main_columns['id']].nunique(), active_data[self.main_columns['id']].nunique()

    def calculate_segment_size(self, node, year_label, non_tail=False):
        df = node.data.loc[node.data[node.main_columns['tail']]==0].copy() if non_tail else node.data.copy()

        total = df.loc[df['year_label_all'] == year_label, self.main_columns['value']].sum()
        return total

    def calculate_brand_size(self,node, year_label, non_tail=False):
        df = node.data.loc[node.data[node.main_columns['tail']]==0].copy() if non_tail else node.data.copy()
        df = df[df['year_label_all'] == year_label].copy()

        brand_total = df.loc[df[self.main_columns['brand']].isin(self.our_brands), self.main_columns['value']].sum()
        return brand_total

    def get_node_path(self):
        path_line = self.get_parent_path_name()
        path_line.append(self.name)
        return path_line

    def find_max_date(self):
        return self.data[self.data[self.main_columns['time']]].max()

    def get_last_n_month_date(self, n_month):
        from pandas.tseries.offsets import DateOffset

        max_date = pd.to_datetime(self.data[self.main_columns['time']].max(), format='%Y-%m')
        start_date = max_date - DateOffset(months = n_month)

        start_date_str = start_date.strftime('%Y-%m')
        return start_date_str
        
    def create_end_node_vtm_input(self,df_prior,  last_n_months, vtm_input_columns):
        
        if self.data.shape[0] >0:
            start_date_str = self.get_last_n_month_date(n_month = last_n_months)
            df_time_limited = self.data.loc[self.data[self.main_columns['time']] > start_date_str,vtm_input_columns].copy()

            df_time_limited = df_time_limited.groupby(self.main_columns['id']).agg({
                self.main_columns['time']:'min',self.main_columns['value']:'sum',
                self.main_columns['unit']:'sum',self.main_columns['scale_weights']:'mean',
                'BRAND':'max', 'VARIANT (SUB-BRAND)':'max', 'MANUFACTURER':'max',
                'LDESC':'max','PRODUCT CLASS':'max', 'EAN':'max' # TODO make it dynamic
                }).reset_index()
            path_str = '/'.join( self.get_node_path()[1:] )

            df_time_limited['end_node'] = path_str

            df_time_limited['is_my_brand'] = 'no'
            df_time_limited.loc[df_time_limited['BRAND'].isin(self.our_brands),'is_my_brand'] = 'yes'

            df_prior = df_prior.append( df_time_limited )

        return df_prior
        
    def prepare_vtm_input(self, df_vtm=pd.DataFrame(), level=0,last_n_months=3, vtm_input_columns=[]):
        vtm_input_columns = list( self.main_columns.values() ) + ['PRODUCT CLASS','BRAND','VARIANT (SUB-BRAND)','MANUFACTURER','LDESC','EAN']
        
        if len(self.list_children())==0:
            df_vtm = self.create_end_node_vtm_input(df_prior= df_vtm, last_n_months=last_n_months, vtm_input_columns=vtm_input_columns)
        else:
            for child in self.children:
                df_vtm = child.prepare_vtm_input(df_vtm, level=level+1, last_n_months=last_n_months, vtm_input_columns=vtm_input_columns )

        if level==0:
            split = df_vtm.end_node.str.split("/",expand=True)
            col_names = ["layer" + str(col) for col in split.columns.to_list() ]
            df_vtm[col_names] = split
        
            self.vtm_input_csv = df_vtm 
        else:
            return df_vtm

    def prepare_vtm_preference(self, pref_dict):
        df = self.vtm_input_csv.copy()
        
        for key in pref_dict:
            df.loc[:,'prefer_'+key] = np.NaN
            for category in pref_dict[key]:
                df.loc[df['end_node'].str.contains(category),'prefer_'+key] = category

        self.vtm_input_csv = df

    def reduce_vtm_input(self, df_vtm, agg_on_list):
        df_my_brand = df_vtm[df_vtm.is_my_brand == 'yes'].copy()
        df_my_brand['agg_level'] = np.NaN

        df_ao = df_vtm[df_vtm.is_my_brand == 'no'].copy()

        
        layers_dict = {col:'max' for col in list(df_ao.filter(regex='layer').columns)}
        pref_dict = {col:'max' for col in list(df_ao.filter(regex='prefer').columns)}
        main_col_dict ={
            self.main_columns['id']:'min',
            self.main_columns['time']:'min',self.main_columns['value']:'sum',
            self.main_columns['unit']:'sum',self.main_columns['scale_weights']:'mean',
            'is_my_brand':'max',
            'LDESC':'min'
        }
        grp_dict = { **main_col_dict, **layers_dict, **pref_dict}

        df_ao = df_ao.groupby(agg_on_list).agg(grp_dict).reset_index()
        df_ao['LDESC'] = 'ex_' + df_ao['LDESC'].astype(str)
        df_ao['agg_level'] = '_'.join( agg_on_list )

        df = df_my_brand.append(df_ao)
        return df

    def filter_vtm_input(self, df, filter_dict): # TDO dynamiclly
        for key in filter_dict:
            df = df[~df[key].isin(filter_dict[key] )]

        end_node = df['end_node']
        df = df.drop(columns=['end_node'] )
        df['end_node'] = end_node

        col_list = list(df.filter(regex='layer').columns) + list(df.filter(regex='prefer').columns) + ['agg_level','PRODUCT CLASS','BRAND','MANUFACTURER','VARIANT (SUB-BRAND)','EAN']
        for col in col_list:
            df[col] = df[col].str.replace(' \(.*\)','',regex=True)
            df[col] = df[col].str.replace('\(.*\)','',regex=True)
            df[col] = df[col].str.replace('&|#|/|\'|!|`|-|\.','',regex=True)
            
        df[self.main_columns['id']] = 'sku' + df[self.main_columns['id']].astype(str)
        df = df.drop(columns=['agg_level'])
        return df

    def save_vtm_input(self,last_n_months=3, vtm_input_columns=[], agg_on_list=[], pref_dict={}, filter_dict={}):
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d_%H%M")

        self.prepare_vtm_input(last_n_months=last_n_months, vtm_input_columns=[])

        pref_dict = {
            'color':['Red_Wine','Pink_Nude','COLOR_AO'],
            # 'brand':['Maybelline','Loreal'],
        }
        self.prepare_vtm_preference(pref_dict=pref_dict)

        agg_on_list = ['end_node','PRODUCT CLASS','BRAND','MANUFACTURER','VARIANT (SUB-BRAND)','EAN']
        df = self.reduce_vtm_input(self.vtm_input_csv,agg_on_list=agg_on_list)

        filter_dict = {'end_node':['class_AO']}
        df = self.filter_vtm_input(df, filter_dict = filter_dict)

        df = self.prepare_periscope_csv(df)

        file_input_xl = f'/vtm/{now}_vtm_input.xlsx'
        df.to_excel(folder+file_input_xl,index=False)

        file_input_csv = f'/vtm/{now}_vtm_input.csv'
        df.to_csv(folder+file_input_csv,index=False)

    def __reduce_by_distribution_weight(self, list_combinations, list_weight, treshold = 0.05):
        reduced_list = [matrix for matrix in list_combinations if self.__is_all_above_treshold(matrix, list_weight, treshold)]
        reduced_list = [matrix for matrix in reduced_list if len(matrix) > 1]

        return reduced_list

    def __is_all_above_treshold(self, matrix, list_weight, treshold):
        for elements in matrix:
            weihgts = [list_weight.iloc[index] for index in elements]
            if sum(weihgts) <= treshold:
                return False
        return True

    ###################### Dıvide the elements into buckets #####################################
    def __pre_partition_reduction(self, attr, aggresive_elimination = False):
        # aggresive_elimination = True if attr.shape[0] > 13 else False
        treshold= 0.01 if aggresive_elimination else 0.001

        attr_name = attr.index.name

        attr = attr.reset_index().sort_values(by='%', ascending=True) # TODO reset'i sil label'ı kopyala ve other gruplamasını yap
        attr['cumsum'] = attr['%'].cumsum()
        attr['%_shift'] = attr['%'].shift(periods=-1)
        attr['is_grp_larger'] =  ~(attr['%_shift'] > attr['cumsum'])

        attr['label'] = attr[attr_name]
        attr.loc[(attr.is_grp_larger == False) & (attr['%'] <= treshold) ,'label'] = 'OTHER'

        attr = attr.set_index(attr_name).drop(columns=['cumsum','%_shift','is_grp_larger'])

        return attr


    def __map_all_partititon(self, len, max_bucket = None): #TODO: Improve this algorithm with Dynamic programming memoization
        import more_itertools as mit

        if max_bucket is None:
            max_bucket = len
            
        l = list(range(len))
        comb = [part for n_bucket in range(1, max_bucket + 1) for part in mit.set_partitions(l, n_bucket)]
        return comb

    def __map_distribution(self, K, N): #TODO: Improve this algorithm with Dynamic programming memoization
        all_combinations = []
        for x in range(N**K):
            t = x
            l = [[] for _ in range(N)]
            for i in range(K):
                id = t % N
                t = t // N   #integer division
                l[id].append(i)
            all_combinations.append(l)

        return all_combinations

    def __reduce_mapdistribution(self, matrix_list):
        matrix_list = [[ x for x in matrix if x] for matrix in matrix_list] # throw emplty elements

        reduced_list = []
        for matrix in matrix_list:
            sorted_matrix = sorted(matrix, key=lambda x: x[0])
            if not sorted_matrix in reduced_list:
                reduced_list.append(sorted_matrix)

        return reduced_list

    ###################### Dıvide the elements into buckets in order############################
    
    def __generate(self, n, l):
        from itertools import product

        for c in product(range(1, l), repeat=n - 1):
            s = sum(c)
            if s > l - 1:
                continue
            yield *c, l - s

    def __ordered_partition_n_bucket(self, length, n_bucket):
        iterable = list(range(length))
        comb = []
        for n in range(1,n_bucket+1):
            for groups in self.__generate(n_bucket, len(iterable)):
                l, out = iterable, []
                for g in groups:
                    out.append(l[:g])
                    l = l[g:]
                comb.append(out)
        return comb

    def crete_featrues_w_cut_off(self, attr_featrues, max_independent_element = 4, n_cut_off_buckes = 3):
        import copy

        n_items = attr_featrues.shape[0]

        attr_name = attr_featrues.index.name
        attr_featrues = attr_featrues.reset_index().sort_values(attr_name, ascending=False).reset_index(drop=True).reset_index().rename(columns={'index':'order_index'}).sort_values('%',ascending=False).set_index(attr_name)

        cutoff_combinations = self.__ordered_partition_n_bucket(n_items,n_cut_off_buckes)
        
        combs = []
        for n_distict in range(max_independent_element+1):
            main_combs =  copy.deepcopy(cutoff_combinations)
            distinct_features = attr_featrues.iloc[0:n_distict].order_index
            for comb in main_combs:
                for item in distinct_features:
                    for grp in comb:
                        if item in grp: grp.remove(item)
                    comb.append([item])
                
                comb = [element for element in comb if element != []]

                combs.append(comb)

        return combs, attr_featrues


    def determine_bucket_size(self, attribute_featrues):
        n_items = attribute_featrues.shape[0]

        if n_items > 13: # Try simple elimination
            aggresive_elimination = False
            attribute_featrues = self.__pre_partition_reduction(attribute_featrues, aggresive_elimination=aggresive_elimination)
            n_items = attribute_featrues['label'].nunique()
        
        if n_items > 13:# Try aggresive elimination
            aggresive_elimination = True
            attribute_featrues = self.__pre_partition_reduction(attribute_featrues, aggresive_elimination=aggresive_elimination)
            n_items = attribute_featrues['label'].nunique()

        is_between = n_items > 13 and n_items <= 15
        is_too_many = n_items > 15
        n_bucket = 4 if is_between else 2 if is_too_many else n_items
        
        return n_items, n_bucket, attribute_featrues

    def form_hypo_dict(self, combinations, attr_featrues):
        attr_name = attr_featrues.index.name
        hypo_list = []
        for idx, comb in enumerate(combinations):
            hypo_name = f'h{idx+1}_{attr_name}'
            

            new_hypo = {}
            for idx_grp, grp in enumerate(comb):
                grp_percent = int( attr_featrues.iloc[grp]["%"].sum()*100 )
                grp_size = len(grp)
                grp_biggest_item = attr_featrues.iloc[grp].iloc[0].name
                grp_features = list( attr_featrues.iloc[grp].index )

                new_hypo.update({f'g{idx_grp+1}p{grp_percent}c{grp_size}_n{grp_biggest_item}':grp_features})

                hypo_name += f'_vs_{grp_biggest_item}_c{grp_size}_p{grp_percent}'

            hypo = {hypo_name:new_hypo}
            hypo_list.append(hypo)

        return hypo_list
        

    def create_automatic_hypothesis(self, children_to_focus = None, addtive = True,treshold=0.05, brute_force_treshold = 300*1000):
        if not addtive: # if you want to start from scratch
            self.hypothesises = []

        if children_to_focus is None:
            children_to_focus = self.possible_children

        print(f'\nCreating automatic hypothesis for {self.get_parent_path_name()} -> {self.name} #######################')
        for potential_child in children_to_focus:
            child_items = self.get_possible_child_items(potential_child)
            if pd.api.types.is_numeric_dtype(child_items.index.dtype): #if child numeric
                index_combination, child_items = self.crete_featrues_w_cut_off(child_items, max_independent_element=4, n_cut_off_buckes=3)
                weight_list = child_items['%'].sort_index(ascending=False)
            else:
                n_items, n_bucket, child_items =  self.determine_bucket_size(child_items)
                index_combination = self.__map_all_partititon(n_items, n_bucket)
                weight_list = child_items['%']

            reduced_combination = self.__reduce_by_distribution_weight(index_combination, weight_list, treshold=treshold)

            auto_hypo = self.form_hypo_dict(reduced_combination, child_items)

            print(f'{len(auto_hypo)} auto hypothesis has been generated from {potential_child}!')
            self.add_hypothesis(addition= auto_hypo )





# %%