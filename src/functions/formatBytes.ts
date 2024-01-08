export function convertSize(sizeInBytes: number) {
    if (sizeInBytes < 1024) {
      return sizeInBytes + " B";
    } else if (sizeInBytes < 1048576) {
      return (sizeInBytes / 1024).toFixed(2) + " KB";
    } else {
      return (sizeInBytes / 1048576).toFixed(2) + " MB";
    }
  }