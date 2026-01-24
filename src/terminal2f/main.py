from matplotlib.pylab import record
import numpy as np
import rerun as rr
import time

def main():
    print("Hello World")
    for x in range(0, 10):
        run_1 = rr.RecordingStream("Testing me")
        run_1.spawn()
        run_1.log("x", rr.TextLog("hello")) 
        time.sleep(5)
    
if __name__ == "__main__":
    main()

