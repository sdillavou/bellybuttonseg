from createdir import createdir, parse_bellybuttonseg_args

if __name__ == "__main__": 
    args = parse_bellybuttonseg_args('Create a new BellyButton project')
    
    if args is None:  # Error in parsing
        raise ValueError("Error: Invalid arguments provided.")
        
    createdir(0, file_path=args['file_path'], folder_name=args['folder_name']) 
