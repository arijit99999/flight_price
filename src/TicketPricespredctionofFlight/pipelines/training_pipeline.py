from src.TicketPricespredctionofFlight.components.data_ingestion import data_ingestion
from src.TicketPricespredctionofFlight.components.data_transformation import data_transformation
import os
import sys
from src.TicketPricespredctionofFlight.logger import logging
from src.TicketPricespredctionofFlight.exception import customexception


obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()

obj2=data_transformation()
train_arr,test_arr=obj2.data_transform_initiated(train_data,test_data)