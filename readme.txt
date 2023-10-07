Folder structure

- 19114479_ProjectFiles/
    ├── readme.txt
    ├── main.py [RUN THIS FILE]
    ├── algorithm.py
    ├── funcions.py
    ├── stats.py
    ├── performance_measure.py
    │  
    ├── Dataset/
    │  	 └── 623/
    │  	 	└── [images.jpg]
    │   └── 684/
    │   		└── [images.jpg]
    │   └── 9730/
    │   		└── [images.jpg]
    │   └── 10917/
    │   		└── [images.jpg]
    │
    ├── GroundTruth/
    │   └── 623/
    │   		└── Mask/
    │   			└── [Generated Mask]
    │   		└── Skyline/
    │   			└── [Generated Skyline Image]
    │   └── 684/
    │   		└── Mask/
    │   			└── [Generated Mask]
    │   		└── Skyline/
    │   			└── [Generated Skyline Image]
    │   └── 9730/
    │   		└── Mask/
    │   			└── [Generated Mask]
    │   		└── Skyline/
    │   			└── [Generated Skyline Image]
    │   └── 10917/
    │   		└── Mask/
    │   			└── [Generated Mask]
    │   		└── Skyline/
    │   			└── [Generated Skyline Image]
    │   └── 623.png
    │   └── 684.png
    │   └── 9730.png
    │   └── 10917.png

1. Make sure the dataset is placed in the correct directory with correct name before running the main.py
2. Uncomment the block of code to run the algorithm on different dataset.
3. It is suggested to run only for a dataset at the same time.
4. After running the main.py, the output is:
	- popout window that display 12 randomly choosen images and their classification result for day and night
	- console that display texts that describe the overall accuracy of algorithm on that dataset
	- mask and skyline image will generated and save in the folder mentioned above

