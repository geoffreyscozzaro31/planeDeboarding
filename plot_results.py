import matplotlib.pyplot as plt
import pandas as pd

def display_missed_pax_strategies(csv_result):
    df = pd.read_csv(csv_result)
    indexes = df[['Seat Allocation', 'Deboarding Strategy']].apply(lambda x: f'{x[0]} - {x[1]}', axis=1)
    values = df['All Missed Pax'].values
    df = pd.DataFrame({'Strategy': indexes, 'Sum Missed Pax': values})
    df.plot(x='Strategy', y='Sum Missed Pax', kind='bar', stacked=True)
    plt.show()


