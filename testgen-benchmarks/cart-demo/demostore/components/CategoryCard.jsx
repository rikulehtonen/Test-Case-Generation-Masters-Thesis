import Link from 'next/link';
import Image from 'next/image';
import styles from '../styles/CategoryCard.module.css';

const CategoryCard = ({ image, name }) => {
  return (
    <Link href={`/category/${name}`}>
      <div className={styles.card}>
        <Image className={styles.image} src={image} fill={true}/>
        <div className={styles.info}>
          <h3>{name}</h3>
          <p>SHOP NOW</p>
        </div>
      </div>
    </Link>
  );
};

export default CategoryCard;
