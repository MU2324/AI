#Association Rule Mining

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori , association_rules
import pandas as pd
from matplotlib import pyplot as plt

transactions = [
    ['orange', 'lemon', 'banana'],
    ['orange', 'coconut'],
    ['lemon', 'coconut'],
    ['orange', 'lemon', 'coconut'],
    ['orange', 'banana'],
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array , columns = te.columns_)

frequent_items = apriori(df , min_support=0.2 , use_colnames=True)

association_rules =  association_rules(frequent_items , metric='confidence' , min_threshold=0.5)

print("Frequent Itemsets:")
print(frequent_items)
print("\n\nAssociation Rules:")
print(association_rules)

plt.barh(range(len(frequent_items)) , frequent_items['support'] , align='center')
plt.yticks(range(len(frequent_items)) , frequent_items['itemsets'].apply(lambda x:','.join(x)))
plt.xlabel('Support')
plt.title('Frequent Items')
plt.show()
