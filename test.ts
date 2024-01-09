import probe from 'probe-image-size';


async function teste () {
    // By URL with options
    const result = await probe('https://img.mangaschan.com/uploads/manga-images/j/jujutsu-kaisen/capitulo-1/7.webp', { rejectUnauthorized: false });
    console.log(result); 
}

teste()