import { useState } from 'react';
import styles from '../styles/UserPage.module.css';  // Import your CSS module

const UserPage = () => {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [message, setMessage] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const { username, password } = formData;

    if (!username || !password) {
      setMessage({ type: 'error', content: 'Both fields are required!' });
      return;
    }

    if (username === 'test@test.fi' && password === 'test') {
      setMessage({ type: 'success', content: 'Successful login!' });
    } else {
      setMessage({ type: 'error', content: 'Invalid username or password!' });
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Login</h2>
      <form onSubmit={handleSubmit} className={styles.form}>
        {message && (
          <p className={message.type === 'error' ? styles.error : styles.success}>
            {message.content}
          </p>
        )}
        <input 
          type="text" 
          name="username" 
          value={formData.username}
          onChange={handleChange} 
          placeholder="Username" 
          className={styles.input}
        />
        <input 
          type="password" 
          name="password" 
          value={formData.password}
          onChange={handleChange} 
          placeholder="Password" 
          className={styles.input}
        />
        <button type="submit" className={styles.button}>Login</button>
      </form>
    </div>
  );
};

export default UserPage;