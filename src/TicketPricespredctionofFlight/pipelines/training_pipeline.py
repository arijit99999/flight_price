from src.TicketPricespredctionofFlight.components.data_ingestion import data_ingestion

import os
import sys
from src.TicketPricespredctionofFlight.logger import logging
from src.TicketPricespredctionofFlight.exception import customexception


obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()