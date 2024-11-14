mkdir build
find HAL/HAL/ -type f -name "*.java" > sources.txt
find SpatialEGT -type f -name "*.java" >> sources.txt
cp HAL/HAL/lib/* lib/
javac -d "build" -cp "lib/*" @sources.txt