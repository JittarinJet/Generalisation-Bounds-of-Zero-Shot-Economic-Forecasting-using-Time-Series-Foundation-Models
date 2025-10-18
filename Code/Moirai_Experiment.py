import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from uni2ts.eval_util.plot import plot_next_multi
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

step_horizon = 1
MODEL = "moirai" 
SIZE = "large" # model size: choose from {'small', 'base', 'large'}
PSZ = "auto"
BSZ = 32
names = ['GDP', 'Primary_Industries', 'Goods_Producing_Industries', 'Services_Industries', 'GDP_quarterly']
START_POINT = [41]
for start_point in START_POINT:
    for name in names:
        path = f'Moirai/uni2ts/datasets_PhD/{name}.csv'
        moirai_df = pd.read_csv(path, index_col=0, parse_dates=True)
        moirai_df = moirai_df.asfreq('Q')

        train_size = len(moirai_df) - step_horizon
        Y_train_df = moirai_df[:train_size]
        Y_test_df = moirai_df[train_size:]
        actual_df = moirai_df
        
        ds = PandasDataset(dict(actual_df))
        end_point = len(moirai_df)
        TEST = len(moirai_df) - start_point
        CTX =  start_point
        PDT = step_horizon # prediction length: any positive integer
        train, test_template = split(
            ds, offset=-TEST
        ) # assign last TEST time steps as test set

        # Construct expanding windows evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=TEST // PDT,  # number of expanding windows evaluation
            distance=PDT,  # number of time steps between each window
        )

        if MODEL == "moirai":
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
        elif MODEL == "moirai-moe":
            model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=16,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )

        predictor = model.create_predictor(batch_size=BSZ)
        forecasts = predictor.predict(test_data.input)
        temp_forecasts = forecasts
        input_it = iter(test_data.input)
        label_it = iter(test_data.label)
        forecast_it = iter(forecasts)
        temp_input_iter = []
        temp_input_iter_value_last = []
        temp_label_iter = []
        temp_label_iter_values = []
        temp_forecast_iter = []
        temp_forecast_iter_samples_values = []
        
        for _input in input_it:
            temp_input_iter.append(_input)
            temp_input_iter_value_last.append(_input['target'][len(_input['target']) - 1])
        temp_input_iter_values = temp_input_iter[len(temp_input_iter)-1]
        for _label in label_it:
            temp_label_iter.append(_label)
            temp_label_iter_values.append(_label.values)
        for _forecast in forecast_it:
            temp_forecast_iter.append(_forecast)
            temp_forecast_iter_samples_values.append(np.median(_forecast.quantile(0.5)))

        pd.DataFrame(temp_input_iter_value_last).to_csv(f'results/all_steps/Moirai_{name}_{start_point}_{end_point}_{MODEL}_{SIZE}.csv', index=False)
        pd.DataFrame(temp_forecast_iter_samples_values).to_csv(f'results/all_steps/Moirai_{name}_{start_point}_{end_point}_{MODEL}_{SIZE}_forecast.csv', index=False)
        
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(actual_df.index, actual_df[name], label='Actual (All Data)', color='#000c66')
        plt.plot(actual_df[CTX:].index, temp_forecast_iter_samples_values, label='Forecast' , color='#CE9DD9')
        plt.tick_params(axis='x', labelrotation=30)
        plt.title(f'Moirai Model: {name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.savefig(f'results/all_steps/{name}_{start_point}_{end_point}_{MODEL}_{SIZE}.png')