files=rjson::fromJSON(file = 'D:/Eye/train_jpg/try/ZocEye.json')
https.file=c()

for(i in 1:length(files)){
  n=length(files[[i]]['Masks'][[1]])
  for (j in 1:n){
    png=files[[i]]['Masks'][[1]][j][[1]]
    https.file<-c(https.file,png)
    }
}

write.table(https.file,'D:/Eye/train_jpg/try/ZocEyepng.txt',col.names = FALSE,row.names = F)

for(png in https.file){
  #dest=paste0('D:/Eye/train_jpg/try/png/',str_split(basename(png),'\\.')[[1]][1],'.png')
  download.file(png)
}