import pandas as pd
import numpy as np

frame1 = pd.DataFrame(np.arange(6).reshape(2,3),
                     index=['first', 'second'],
                     columns=['one', 'two', 'three'])
print(frame1)

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

print('\n',pd.DataFrame(data, columns=['year', 'state', 'pop']))

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop'],
                      index=['one', 'two', 'three', 'four','five', 'six'])
print('\n',frame2)


print('\n',frame2.columns)

print('\n',frame2['state'])

print('\n',frame2.year) #frame2['year']

print('\n',frame2.loc['three'])






