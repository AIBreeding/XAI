# -*- coding: utf-8 -*-
import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import AutoHPO
AutoHPO.run()
