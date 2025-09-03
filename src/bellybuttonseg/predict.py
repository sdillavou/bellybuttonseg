from createdir import runBB, parse_bellybuttonseg_args

if __name__ == "__main__": 
    args = parse_bellybuttonseg_args('Run BellyButton prediction on specified file/directory')
    runBB(train=False, predict=True, file_path=args['file_path']) 

