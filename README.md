# FastPark : A decentralized camera-based smart parking system employing deep learning for outdoor environments
------------------

## Requirements

 -  Python v3.4
 -  Tensorflow (recommend v1.5.0)
 -  Keras (recommend v2.1.4)

## Steps to reproduce experiments

 1. Clone this repository :

    ```bash
    git clone --recursive https://github.com/Chaiyaboon-Sruayiam/fastpark.git
    ```

 2. Download the datasets using the following links and extract them somewhere

    | Dataset | Link | Size | 
    | ------- | ---- | ---: |
    | SWUPark | https://drive.google.com/open?id=1WpB3YlcAskGdzJofWESCMv3IybRWIadr | 64.0 MB |

 3. Extract SWUPark dataset zip file
    ```bash
    unzip SWUPark.zip
    ```
 4. Edit lines inside train_script.py and inside the code you need to fill the directories before execute this code
    ```bash
    train_labels_path = "<where is your train_label.txt>"
    val_labels_path = "<where is your validate_label.txt>"
    train_root_images_folder = "<root of train image set directory>"
    val_root_images_folder = "<root of validate image set directory>"
    weight_output_filename = "<your weight output name>.h5"
    ```
 5. Execute your train python script
    ```bash
    #train script
    py train_script.py
    #and then you will get the weight (.h5) file.
    ```

 6. Edit lines inside test_script.py and inside the code you need to fill the directories before execute this code
    ```bash
    labels_path = "<where is your test_label.txt>"
    root_images_folder = "<root of test image set directory>"
    weights_file = "<where is your weight_file.h5>"
    text_output_filename ="<your text output as a result file name>"
    ```

7. Execute your test python script
    ```bash
    #test script
    py test_script.py
    #and then you will get the text as a result (.final).
    #you can open .final file by common text editor.
    ```
