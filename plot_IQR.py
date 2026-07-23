import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
#This is the code that I will use to transform my CSVs into plots

# Load csv file, change name/path as needed
df = pd.read_csv('default.csv') 
#these are the values we care about plotting
values = [
  'hpv_prevalence'
]
print(df)
def plot_iqr(value: str, time: str = 't', title = 'IQR timeseries'): #time_col can be 'year' instead
  # Compute IQR + median across runs at each time t, we will use time steps instead of year for now
  q = (
      df.groupby(time)[value]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: 'q25', 0.5: 'median', 0.75: 'q75'})
        .sort_index()
        .reset_index()
  )

  # Convert to numpy arrays 
  x   = q['year'].to_numpy()
  #min = q['min'].to_numpy()
  q25 = q['q25'].to_numpy()
  med = q['median'].to_numpy()
  q75 = q['q75'].to_numpy()
  #max = q['max'].to_numpy()

  # Plot
  fig, ax = plt.subplots()

  # IQR (maybe full range) shaded region
  #ax.fill_between(x, color = 'gray', alpha = 0.2, label = 'Full Range')
  ax.fill_between(x, q25, q75, color = 'lightblue', alpha = 0.3, label = 'IQR')

  # IQR + Median Line
  ax.plot(x, q25, linewidth = 1, color = 'lightblue',)
  ax.plot(x, med, linewidth = 2, label = 'Median', color = 'black')      # prominent
  ax.plot(x, q75, linewidth = 1, color = 'lightblue')
  ax.ticklabel_format(style="plain", axis="y")
  ax.set_xlim(left=1980, right=2050)
  if value.endswith("prevalence"):
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
  else:
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
  #x axis label should be Years Past most of the time, but again incase we want the year instead
  if time == 't':
    ax.set_xlabel('Years Past')
  else:
    ax.set_xlabel(time)
  #plt.axhline(y= 4, color='r', linestyle='-', label = '2040 Target')  
  ax.set_ylim(bottom = 0)
  ax.set_ylabel(value)
  ax.set_title(title)
  ax.grid(True, linewidth = 0.5, alpha = 0.4)
  ax.legend()
  return fig, ax

figs = []
for v in values:
  fig, ax = plot_iqr(value = v, time = 'year', title = f"{v}")
  figs.append(fig)
plt.tight_layout()
plt.show()

