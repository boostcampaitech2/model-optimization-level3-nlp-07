from nvidia.dali.plugin.pytorch import DALIGenericIterator

class DALIDataloader(DALIGenericIterator):
    
    def __init__(self, size,pipeline,  batch_size, output_map=["image", "label"], auto_reset=True, onehot_label=False):
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)
        self._size=size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        

    def __next__(self):
        
        data = super().__next__()[0]
        
        if self.onehot_label:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]]]
        
    def __len__(self):
        
        if self.size%self.batch_size==0: 
            return self.size//self.batch_size
        
        else:
            return self.size//self.batch_size+1