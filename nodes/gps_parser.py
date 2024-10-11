import rospy
import serial
import re
from std_msgs.msg import Float32
from sensor_msgs.msg import NavSatFix

SERIAL_PORT = '/dev/simpleRTK3B'  # Custom USB port
BAUD_RATE = 115200  

# ROS publishers
gps_pub = None
heading_pub = None


def convert_to_decimal(degree_min, direction):
    """
    Converts latitude/longitude from degree-minute format to decimal degrees.
    """
    if not degree_min or not direction:
        return None
    
    degrees = float(degree_min[:2])
    minutes = float(degree_min[2:])
    decimal = degrees + (minutes / 60)

    if direction == 'S' or direction == 'W':
        decimal = -decimal
    
    return decimal


def parse_gga(sentence):
    """
    Parses the GGA sentence to extract latitude, longitude, and altitude.
    """
    fields = sentence.split(',')
    
    if len(fields) < 15:
        rospy.logwarn("Invalid GGA sentence")
        return None
    
    time_utc = fields[1]
    latitude = fields[2]
    lat_direction = fields[3]
    longitude = fields[4]
    lon_direction = fields[5]
    fix_quality = fields[6]
    num_satellites = fields[7]
    altitude = fields[9]

    latitude = convert_to_decimal(latitude, lat_direction)
    longitude = convert_to_decimal(longitude, lon_direction)

    return latitude, longitude, float(altitude)


def parse_uniheadinga(sentence):
    """
    Parses the #UNIHEADINGA message to extract the heading information.
    """
    pattern = r'#UNIHEADINGA,(\d+),GPS,(\w+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+);SOL_COMPUTED,(\w+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),\"(\d+)\",(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\w+)\*([a-fA-F0-9]+)'
    
    match = re.match(pattern, sentence)
    
    if match:
        heading = float(match.group(10))  # Heading in degrees
        return heading
    else:
        rospy.logwarn("Invalid or unmatched UNIHEADINGA sentence")
        return None


def publish_gps(latitude, longitude, altitude):
    """
    Publishes the position as a NavSatFix message.
    """
    navsat_msg = NavSatFix()
    navsat_msg.latitude = latitude
    navsat_msg.longitude = longitude
    navsat_msg.altitude = altitude
    navsat_msg.status.status = 0
    navsat_msg.status.service = 1
    navsat_msg.header.stamp = rospy.Time.now()
    navsat_msg.header.frame_id = 'gps'
    gps_pub.publish(navsat_msg)


def publish_heading(heading):
    """
    Publishes the heading as a Float32 message.
    """
    heading_msg = Float32()
    heading_msg.data = heading
    heading_pub.publish(heading_msg)


def read_serial_data():
    # Open the serial port
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        while not rospy.is_shutdown():
            try:
                # Read a line from the serial port
                line = ser.readline().decode('ascii', errors='ignore').strip()
                
                if line.startswith('$GNGGA'):
                    # Parse GGA message (Position Fix Data)
                    gps_data = parse_gga(line)
                    if gps_data:
                        latitude, longitude, altitude = gps_data
                        publish_gps(latitude, longitude, altitude)
                
                elif line.startswith('#UNIHEADINGA'):
                    # Parse UNIHEADINGA message (Heading Info)
                    heading = parse_uniheadinga(line)
                    if heading:
                        publish_heading(heading)
                
            except serial.SerialException as e:
                rospy.logerr(f"Serial Exception: {e}")
            except Exception as e:
                rospy.logerr(f"Error: {e}")


if __name__ == "__main__":
    rospy.init_node('gps_heading_parser', anonymous=True)
    
    # ROS publishers
    gps_pub = rospy.Publisher('/gps/fix', NavSatFix, queue_size=10)
    heading_pub = rospy.Publisher('/gps/heading', Float32, queue_size=10)
    
    rospy.loginfo("Starting GPS and heading parser node...")
    
    try:
        read_serial_data()
    except rospy.ROSInterruptException:
        pass
