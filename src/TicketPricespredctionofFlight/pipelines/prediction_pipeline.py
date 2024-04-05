import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.TicketPricespredctionofFlight.logger import logging
from src.TicketPricespredctionofFlight.exception import customexception
from src.TicketPricespredctionofFlight.utlis.utils import load_object

class model_pred_config:
    preprcessor_path=os.path.join('artifacts',"preprocessor.pkl")
    model_path=os.path.join('artifacts',"model.pkl")
class model_prediction:
    def __init__(self):
        self.model_pred=model_pred_config()
    def model_pred_initiate(self,features):
        try:
            preprocessor=load_object(self.model_pred.preprcessor_path)
            model=load_object(self.model_pred.model_path)
            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)

class custom_data:
    def __init__(self,airline,source_city,departure_time,stops,arrival_time,destination_city,classes,duration,days_left):
        self.airline=airline
        self.source_city=source_city
        self.departure_time=departure_time
        self.stops=stops
        self.arrival_time=arrival_time
        self.destination_city=destination_city
        self.classes=classes
        self.duration=duration
        self.days_left=days_left

    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'airline':[self.airline],
                    'source_city':[self.source_city],
                    'departure_time':[self.departure_time],
                    'stops':[self.stops],
                    'arrival_time':[self.arrival_time],
                    'destination_city':[self.destination_city],
                    'classes':[self.classes],
                    'duration':[self.duration],
                    'days_left':[self.days_left]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
    