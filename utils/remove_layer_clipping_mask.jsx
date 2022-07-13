https://community.adobe.com/t5/photoshop-ecosystem-discussions/list-with-all-files-in-the-folder/td-p/10549286
main();

function main(){
    // pop out a dialog to select the source image folder
    var selectedFolder = Folder.selectDialog("Please select folder contains psd files");
    if(selectedFolder == null) return;
    var fileList= selectedFolder.getFiles("*.psd");

    // pop out a dialog to select the output folder
    var outFolder = Folder.selectDialog("Please select the output folder");
    if(outFolder == null) return;

    // remove the clipping mask of each layer if possible
    if(fileList.length>0){
        for (i =0; i < fileList.length; i++){
            // open the psd
            var fileRef = new File(fileList[i]);
            var docRef = app.open(fileRef);
            // iterate each layers
            clipScan(docRef);
            }

        }
    };

// thanks to https://community.adobe.com/t5/illustrator-discussions/remove-clipping-mask/m-p/1161937
function clipScan (docRef) {

    for (i=docRef.pageItems.length-1;i>=0;i--) { 
        if (docRef.pageItems.clipping == true){
            docRef.pageItems.remove();
        }
    }

};