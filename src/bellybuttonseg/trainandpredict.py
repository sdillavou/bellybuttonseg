from createdir import runBB, parse_bellybuttonseg_args

if __name__ == "__main__": 
    args = parse_bellybuttonseg_args('Run BellyButton training and prediction on specified file/directory')
    runBB(train=True, predict=True, file_path=args['file_path']) 

