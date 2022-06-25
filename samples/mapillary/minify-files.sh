local resize_factor=50%

find . -iname '*.jpg' -exec convert \{} -verbose -resize $resize_factor \{} \;
find . -iname '*.png' -exec convert \{} -verbose -resize $resize_factor \{} \;