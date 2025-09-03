from createdir import createdir, parse_bellybuttonseg_args

if __name__ == "__main__": 
    args = parse_bellybuttonseg_args('Create BellyButton example 3 (Spacefoam) project in specified directory')
    createdir(3, file_path=args['file_path'], folder_name=args['folder_name']) 

