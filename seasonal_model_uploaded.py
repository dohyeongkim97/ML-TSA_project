import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
import ast
import statsmodels.tsa.api as tsa

scaler = StandardScaler()

df = pd.read_csv("/home/ubuntu/airflow/data/update/cat1_update.csv")


# 12월 31일은 주마다 주기가 안맞아서 패스(7*52 = 364)
if (datetime.now().day == 31) and (datetime.now().month==12):
  print('')

else:
  # 특정 요일(월요일) 주기로 모델 리뉴얼
  if datetime.now().weekday() == 3:
    indexes = []

    arima_params = pd.read_csv("/home/ubuntu/airflow/ML/arima_parameter_dataset.csv")
    arima_halfyear_params = pd.read_csv("/home/ubuntu/airflow/ML/arima_parameter_dataset_halfyear.csv")
    fitting_params = pd.read_csv("/home/ubuntu/airflow/ML/fitting_parameter.csv")

    forecast_annual = pd.DataFrame(columns = arima_params['catlist'])
    forecast_halfyear = pd.DataFrame(columns = arima_halfyear_params['catlist'])

    #7일마다 날짜 떼기
    for i in range(len(df.index)):
      if (i+1) % 7 == 0:
        indexes.append(df.index[i])
        print(i)

    # df를 7일 주기로 잘라내기.
    df_idx = df.loc[indexes, :]

    df_idx = df_idx.set_index('period')



    # 학습(1년 주기)
    for i in range(len(arima_params)):
      best_model = tsa.statespace.SARIMAX(df_idx[arima_params.loc[i, 'catlist']],
                                          order = ast.literal_eval(arima_params.loc[i, 'paramlist']),
                                          seasonal_order = ast.literal_eval(arima_params.loc[i, 'seasonal']),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
      best_result = best_model.fit()

      forecast = pd.DataFrame(best_result.forecast(30))

      forecast_annual[arima_params['catlist'][i]] = forecast

    #     with open(f'./arima_models/arima_{arima_params.loc[i, "catlist"][:2]}.pkl', 'wb') as model_file:
    #         pickle.dump(best_model, model_file)

    # 리뉴얼된 모델 저장
      with open(f'/home/ubuntu/airflow/ML/arima_results/arima_{arima_params.loc[i, "catlist"][:2]}_result.pkl', 'wb') as model_file:
            pickle.dump(best_result, model_file)

    # 학습(반년 주기)
    for i in range(len(arima_halfyear_params)):
      best_model_halfyear = tsa.statespace.SARIMAX(df_idx[arima_halfyear_params.loc[i, 'catlist']],
                                          order = ast.literal_eval(arima_halfyear_params.loc[i, 'paramlist']),
                                          seasonal_order = ast.literal_eval(arima_halfyear_params.loc[i, 'seasonal']),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
      best_result_halfyear = best_model_halfyear.fit()

      forecast_half = pd.DataFrame(best_result_halfyear.forecast(30))


      forecast_halfyear[arima_halfyear_params['catlist'][i]] = forecast_half

    #     with open(f'./arima_models/arima_{arima_params.loc[i, "catlist"][:2]}.pkl', 'wb') as model_file:
    #         pickle.dump(best_model, model_file)

    # 리뉴얼된 모델 저장
      with open(f'/home/ubuntu/airflow/ML/arima_halfyear_results/arima_{arima_halfyear_params.loc[i, "catlist"][:2]}_result.pkl', 'wb') as model_file:
            pickle.dump(best_result, model_file)

    # 시작 날짜
    start_date = pd.to_datetime(str(df['period'][len(df)-1])[:10])

    # 날짜 간격
    delta = timedelta(days=1)  # 1일 간격으로 설정

    # 날짜 리스트 초기화
    date_list = []

    # 210일간의 날짜 리스트 생성
    for i in range(210):
      current_date = start_date + i * delta
      date_list.append(current_date)

    # 연간 모델 예측 데이터를 담을 DF 생성(인덱스 : 매일 날짜, 7*30주)
    predicted_timeseries = pd.DataFrame(index = date_list)
    predicted_timeseries_halfyear = pd.DataFrame(index = date_list)

    # 인덱스 기준, 연간 모델 예측 데이터 통합(merge, nan값 살려서 통합. interpolate 하기 위함)
    predicted_timeseries = pd.merge(predicted_timeseries,
                                    forecast_annual,
                                    left_on = predicted_timeseries.index,
                                    right_on = forecast_annual.index,
                                    how='left')


    # 반년 모델 예측 데이터를 담을 DF 생성(인덱스 : 매일 날짜, 7*30주)
    predicted_timeseries_halfyear = pd.merge(predicted_timeseries_halfyear,
                                              forecast_halfyear,
                                              left_on = predicted_timeseries_halfyear.index,
                                              right_on = forecast_half.index,
                                              how = 'left')

    # 각 데이터프레임 컬럼 카테고리명으로 설정
    predicted_timeseries = predicted_timeseries.set_index('key_0')
    predicted_timeseries_halfyear = predicted_timeseries_halfyear.set_index('key_0')
    predicted_timeseries.columns = arima_params['catlist']
    predicted_timeseries_halfyear.columns = arima_params['catlist']

    # 원본 데이터의 마지막 데이터(당일 데이터)를 0번 행에 기입. interpolate하기 위함.
    predicted_timeseries.iloc[0, :] = df.iloc[len(df)-1, range(1, len(df.columns))]
    predicted_timeseries_halfyear.iloc[0, :] = df.iloc[len(df)-1, range(1, len(df.columns))]

    #         predicted_timeseries.columns = arima_params['catlist']
    #         predicted_timeseries_halfyear.columns = arima_params['catlist']

    # 타입 float로 변환
    predicted_timeseries = predicted_timeseries.astype(float)
    predicted_timeseries_halfyear = predicted_timeseries_halfyear.astype(float)

    # interpolation
    predicted_timeseries = predicted_timeseries.interpolate()
    predicted_timeseries_halfyear = predicted_timeseries.interpolate()

    # 최종 데이터 DF
    final_prediction = pd.DataFrame(index = date_list, columns = arima_params['catlist'])

    # 각 모델 예측(연간/반년)에 미리 구해놓은 적합값 맞춰서 곱셈.
    # scaler는 Standard Scaler 사용. 각 모델 예측값의 평균과 변동값을 분리하여 계산하기 위함.
    # 예측값에 대하여 변동값 가중치와 평균값(상수) 가중치를 더하여 계산함.
    for i in range(len(arima_params['catlist'])):
      final_prediction[arima_params['catlist'][i]] = scaler.fit_transform(np.array(predicted_timeseries[arima_params['catlist'][i]]*int(fitting_params.loc[i, '1'])/10
                                                                                  +predicted_timeseries_halfyear[arima_params['catlist'][i]]*int(fitting_params.loc[i, '2'])/10).reshape(-1, 1))*int(fitting_params.loc[i, '3'])+np.mean(predicted_timeseries[arima_params['catlist'][i]]*int(fitting_params.loc[i, '4'])/10 + predicted_timeseries_halfyear[arima_params['catlist'][i]]
                                        *(int(fitting_params.loc[i, '5'])/10))

    final_prediction.to_csv('/home/ubuntu/airflow/ML/predictions/seasonal_results.csv')

  else:
    pass

